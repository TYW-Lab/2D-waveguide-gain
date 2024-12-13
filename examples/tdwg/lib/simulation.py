import torch
import torch.fft as fft
from fft_conv_pytorch import fft_conv, FFTConv2d
import numpy as np
from scipy.stats import multivariate_normal
import tdwg.lib.ftutils_np as ftutils
import matplotlib.pyplot as plt
from tdwg.lib.DMD_patterns import generate_fill_factor_mask
from tdwg.lib.ftutils_torch import fft_centered_ortho, ft_f_axis, ifft_centered_ortho
import astropy.units as u
import astropy.constants as const
import copy
from scipy.interpolate import interp1d, interp2d
from tdwg.lib.DMD_patterns import invert_pattern

import astropy.constants as const
import pprint
from dataclasses import dataclass, fields
import sympy as sp


class WaveguideSimulation():
    def __init__(self, neff, Lx, Lz, Nx, Nz, diffusion_length, Ncom=1, I_sat=float('inf'), thickness = 1*u.um, material = None, current = None):
        # all the units are in astropy.units (u)
        #all the units for length are in microns!
        self.n =  neff # effective refractive index for the slab waveguide
        self.Lx = Lx # length of transverse direction
        self.Lz = Lz # beam propagation distance
        
        self.Nx = Nx #Be power of 2 for FFT
        self.Nz = Nz
        
        self.lam0 = 1.55*u.um #wavelength of the fundamental
        
        self.Ncom = Ncom # wavefront is saved every Ncom integration steps


        self.k0 = 2*np.pi/self.lam0 # k-number in free space

        self.dx = self.Lx/(self.Nx-1) # stepsize in transverse dimension
        
        self.x_axis = ftutils.ft_t_axis(self.Nx, self.dx)
        self.z_axis = np.linspace(0, self.Lz, self.Nz) # array holding z-coordinates in um

        self.dz = self.z_axis[1]-self.z_axis[0] # stepsize in propagation direction

        if material is None:
            self.material = Material(dx = self.dx, dz = self.dz, thickness=thickness)
        else:
            self.material = material

        if current is None:
            self.current = torch.from_numpy(np.zeros((self.Nz, self.Nx), dtype=np.float32))
        else:
            self.set_current(current)

        self.set_delta_n(torch.from_numpy(np.zeros((self.Nz, self.Nx), dtype=np.complex128)))
        self.delta_n_active = torch.from_numpy(np.zeros((self.Nz, self.Nx), dtype=np.complex128))
        
        
        self.fx_axis = ft_f_axis(self.Nx, self.dx)
        self.kx_axis = 2*np.pi*self.fx_axis

        # The following defines *dimensionless* quantities, which are used in the simulation internal loop!
        difr_list = np.fft.fftshift(np.exp((-1j*self.kx_axis**2/(2*self.n*self.k0)*self.dz).decompose().value))
        self.difr_list = torch.tensor(difr_list)
        self.k0dz = (self.k0*self.dz).decompose().value

        self.set_diffusion_length(diffusion_length)

        self.set_I_sat(I_sat)

        self.x2ind = lambda x: np.argmin(np.abs(self.x_axis-x))
        self.z2ind = lambda z: np.argmin(np.abs(self.z_axis-z))
        self.zlist2ind = lambda z: np.argmin(np.abs(self.z_list-z))

        # create multivariate gaussian to emulate carrier diffusion
        self.kernel = torch.FloatTensor(get_gaussian_kernel(6*self.nz_kernel+1, 6*self.nx_kernel+1)).unsqueeze(0).unsqueeze(0)

    def set_diffusion_length(self, diffusion_length):
        self.diffusion_length = diffusion_length
        self.nx_kernel = int(np.round((self.diffusion_length/self.dx).decompose()))
        self.nz_kernel = int(np.round((self.diffusion_length/self.dz).decompose()))

    def set_delta_n(self, delta_n):
        if delta_n.shape != (self.Nz, self.Nx):
            raise ValueError('spatial_map has wrong shape, should be [self.Nz, self.Nx]')
        self.delta_n = delta_n
        self.delta_n_SS = delta_n.clone() # delta_n at steady state with input light

    def set_current(self, current):
        if current.shape != (self.Nz, self.Nx):
            raise ValueError('spatial_map has wrong shape, should be [self.Nz, self.Nx]')
        self.current = current

    def set_I_sat(self, I_sat):
        # Set saturation intensity to I_sat (deifined as E^2 for simplicity)
        self.I_sat = I_sat

    def update_delta_n_slice(self, delta_n_slice, a):
        if self.I_sat != np.inf:
            R = torch.abs(a)**2 / self.I_sat
            n_i_new = torch.imag(delta_n_slice) / (1 + R) # Marculescu Eqn. 2.22
            n_i_new = torch.imag(delta_n_slice) / (1 + R * torch.exp(-self.k0dz * n_i_new)) # Correction to account for grid mismatch
            delta_n_slice.copy_(torch.real(delta_n_slice) + 1j * n_i_new)
            return torch.real(delta_n_slice) + 1j * n_i_new
        else:
            return delta_n_slice

    def get_delta_n_active_slice(self, current_slice, a):
        # Parameters: current_slice at z+dz/2 and light field a at z
        # Returns: delta_n at z+dz/2
        material = self.material
        optical_power = np.abs(a)**2
        delta_n_i_at_z = np.imag(material.update(current_slice, optical_power, update_delta_n_r = False)) # approximate value of n_i at z
        delta_n = material.update(current_slice, optical_power * np.exp(- self.k0dz * delta_n_i_at_z))
        return torch.from_numpy(delta_n.astype(np.complex128))
        

    def smoothen_spatial_map(self, spatial_map, padding_mode = 'reflect'):
        # convolves a spatial map (such as a delta_n) with a gaussian kernel of standard deviation equal to the carrier diffusion length (in um). The input should have size of [self.Nz, self.Nx]

        if spatial_map.shape != (self.Nz, self.Nx):
            raise ValueError('spatial_map has wrong shape, should be [self.Nz, self.Nx]')

        # pad delta n and convolve it with gaussian kernel
        spatial_map = fft_conv(spatial_map.unsqueeze(0).unsqueeze(0), self.kernel.to(spatial_map.device), bias=None, padding = [3*self.nz_kernel,3*self.nx_kernel], padding_mode=padding_mode)
        return spatial_map.squeeze(0).squeeze(0)

    def run_simulation(self, a, delta_n = None):
        """
        Use this if you want code to be fast!
        a: The input beam!
        """
        if delta_n is None: delta_n_SS = self.delta_n_SS.to(a.device)
        else: 
            self.set_delta_n(delta_n.to(a.device))
            delta_n_SS = self.delta_n_SS
            
        difr_list = self.difr_list.to(a.device)

        for (z_ind, delta_n_slice) in enumerate(delta_n_SS):
            self.update_delta_n_slice(delta_n_slice, a)
            a = torch.exp(1j*self.k0dz*delta_n_slice) * a
            ak = fft.fft(a)
            ak = difr_list * ak
            a = fft.ifft(ak)
        return a
    
    def run_simulation_slow(self, a, delta_n = None):
        """
        Around 2X slower than the fast version for Nz being 1000
        """
        self.a_list = []
        self.ak_list = []
        self.z_list = []

        if delta_n is None: delta_n_SS = self.delta_n_SS.to(a.device)
        else: 
            self.set_delta_n(delta_n.to(a.device))
            delta_n_SS = self.delta_n_SS
        
        z = 0*u.um
        current = self.current
        delta_n_active = self.delta_n_active
        for z_ind, (delta_n_slice, delta_n_static_slice, current_slice) in enumerate(zip(delta_n_SS, self.delta_n, current)):
            deta_n_slice = self.update_delta_n_slice(delta_n_slice, a)
            delta_n_active_slice = self.get_delta_n_active_slice(current_slice, a)
            #delta_n_active_slice = 0
            delta_n_active[z_ind] = delta_n_active_slice
            total_delta_n_slice = deta_n_slice + delta_n_static_slice + delta_n_active_slice
            a = torch.exp(1j*self.k0dz*total_delta_n_slice) * a
            ak = fft.fft(a)
            ak = self.difr_list.to(a.device) * ak
            a = fft.ifft(ak)

            if z_ind % self.Ncom == 0:
                self.a_list.append(a)
                self.ak_list.append(torch.fft.ifftshift(ak))
                self.z_list.append(copy.deepcopy(z))
            z += self.dz

        '''
        for (z_ind, delta_n_slice) in enumerate(delta_n_SS):
            self.update_delta_n_slice(delta_n_slice, a)
            a = torch.exp(1j*self.k0dz*delta_n_slice) * a
            ak = fft.fft(a)
            ak = self.difr_list.to(a.device) * ak
            a = fft.ifft(ak)

            if z_ind % self.Ncom == 0:
                self.a_list.append(a)
                self.ak_list.append(torch.fft.ifftshift(ak))
                self.z_list.append(copy.deepcopy(z))
            z += self.dz
        '''
        self.Emat_x = torch.stack(self.a_list)
        self.Emat_f = torch.stack(self.ak_list)
        self.z_list = u.Quantity(self.z_list)

        self.Eout_x = self.Emat_x[-1].detach().cpu().numpy()
        self.Eout_f = self.Emat_f[-1].detach().cpu().numpy()
        self.Iout_x = abs(self.Emat_x[-1].detach().cpu().numpy())**2
        self.Iout_f = abs(self.Emat_f[-1].detach().cpu().numpy())**2

        self.Ein_x = self.Emat_x[0].detach().cpu().numpy()
        self.Ein_f = self.Emat_f[0].detach().cpu().numpy()
        self.Iin_x = abs(self.Emat_x[0].detach().cpu().numpy())**2
        self.Iin_f = abs(self.Emat_f[0].detach().cpu().numpy())**2
        return a

    def new_wg_Nz(self, Nz_new, Ncom=1):
        z_axis_new = np.linspace(self.z_axis[0], self.z_axis[-1], Nz_new)
        wg_new = WaveguideSimulation(self.n, self.Lx, self.Lz, self.Nx, Nz_new, Ncom=Ncom)
            
        wg = self
        interp_func = interp2d(wg.x_axis.to("um").value, wg.z_axis.to("um").value, wg.delta_n.cpu().detach().numpy(), kind="linear", bounds_error=False, fill_value=0)
        delta_n_new = interp_func(wg.x_axis.to("um").value, z_axis_new.to("um").value)
        delta_n_new = torch.tensor(delta_n_new)
        wg_new.set_delta_n(delta_n_new)
        return wg_new

    def _plot_delta_n(self, xlim=200, steady_state = False, real = True):
        wg = self
        if steady_state: delta_n = wg.delta_n_SS + wg.delta_n_active
        else: delta_n = wg.delta_n
        delta_n = delta_n.to(dtype=torch.complex128)
        if real: data = delta_n.real
        else: data = delta_n.imag
        plt.pcolormesh(wg.z_axis.to("mm").value, wg.x_axis.to("um").value, data.T.detach().cpu()*1e3, cmap="binary", shading="auto")
        plt.colorbar()
        plt.ylabel("x (um)")
        plt.xlabel("z (mm)")
        plt.ylim(-xlim, xlim)
        #plt.gca().invert_xaxis()

        state = ["Original","Steady-state"][steady_state]
        real_str = ["imag","real"][real]
        plt.title(fr"{state} $\Delta n_{{{real_str}}}\ \  (10^{{-3}})$")
        plt.grid(alpha=0.5)

    
    def _plot_Imat_x(self, xlim=200, renorm_flag=True):
        wg = self
        Emat_x = wg.Emat_x.detach().cpu().numpy()
        Imat_x = np.abs(Emat_x)**2

        if renorm_flag:
            Imat_x = (Imat_x.T/np.max(Imat_x, axis=1)).T


        plt.pcolormesh(wg.z_list.to("mm").value, wg.x_axis.to("um").value, Imat_x.T, cmap="inferno", vmin=0, shading="auto")
        plt.xlabel("z (mm)")
        plt.ylabel("x (um)")
        plt.ylim(-xlim, xlim)
        plt.colorbar()
        #plt.gca().invert_xaxis()
        plt.title("Spatial intensity")
        plt.grid(alpha=0.5)

    def _plot_Imat_f(self, flim=80, renorm_flag=True):
        wg = self
        Emat_f = wg.Emat_f.detach().cpu().numpy()
        Imat_f = np.abs(Emat_f)**2

        if renorm_flag:
            Imat_f = (Imat_f.T/np.max(Imat_f, axis=1)).T

        plt.pcolormesh(wg.z_list.to("mm").value, wg.fx_axis.to("1/mm").value, Imat_f.T, cmap="inferno", vmin=0, shading="auto")
        plt.xlabel("z (mm)")
        plt.ylabel("f (1/mm)")
        plt.ylim(-flim, flim)
        plt.colorbar()
        #plt.gca().invert_xaxis()
        plt.title("Wavevector intensity")
        plt.grid(alpha=0.5)

    def plot_mats(self, xlim=200, flim=80, renorm_flag=True):
        """
        renorm_flag: If true, the plots are renormalized to the maximum value of the intensity for a given zaxis point.
        """        
        wg = self #to save rewriting the code!

        fig, axs = plt.subplots(3, 2, figsize=(10, 6))
        fig.subplots_adjust(wspace=0.4)

        plt.sca(axs.flatten()[0])
        self._plot_delta_n(xlim=xlim,steady_state=False,real=True)

        plt.sca(axs.flatten()[1])
        self._plot_delta_n(xlim=xlim,steady_state=False,real=False)

        plt.sca(axs.flatten()[2])
        self._plot_delta_n(xlim=xlim,steady_state=True,real=True)

        plt.sca(axs.flatten()[3])
        self._plot_delta_n(xlim=xlim,steady_state=True,real=False)

        plt.sca(axs.flatten()[4])
        self._plot_Imat_x(xlim=xlim, renorm_flag=renorm_flag)

        plt.sca(axs.flatten()[5])
        self._plot_Imat_f(flim=flim, renorm_flag=renorm_flag)

        for ax in axs.flatten():
            plt.sca(ax)
        
        plt.tight_layout()
        return fig, axs

########### Material class ##########

@dataclass
class Material:
    '''
    Carrier rate equation model: dN/dt = injection_rate + spontaneous_decay_rate + stimulated_decay_rate
    Reference: 
        [1] Semiconductors on NSM, https://www.ioffe.ru/SVA/NSM/Semicond/GaInAsP/index.html
        [2] Material gain of bulk 1.55 mm InGaAsP/InP semiconductor optical amplifiers approximated by a polynomial model,
            https://doi.org/10.1063/1.371909
        [3] Diode Lasers and Photonic Integrated Circuits, https://onlinelibrary.wiley.com/doi/book/10.1002/9781118148167
        [4] Semiconductor Optical Amplifiers Modeling, Signal Regeneration and Conversion, 
            https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/272998/1/20180627_Thesis_Marculescu_Book_noCV.pdf
    injection_rate = current/(e*V)
        current: injected current
        e: electrical charge
        V = core thickness * dx * dz: volume of active region
            dx, dz: simulation grid sizes in the x and z directions, respectively
    spontaneous_decay_rate = (N-N0)*(A + B*(N-N0) + C*(N-N0)**2)
        N0: intrinsic carrier density, or steady state N when current, P = 0 (4.4e9 cm^(-3) (for y=0.47); 6.7e11 cm^(-3) (for y=1.0))
            I included N0 to account for thermal generation. Although I'm not sure if the form is the most appropriate.
            N0 is really small compared to any N when current is injected (~1e18 cm^(-3)) so probably don't really need to be considered
        A: recombination at defects and surfaces
        B: Radiative recombination coefficient = bimolecular recombination coefficient (e.g. 1.1e-10 cm^3/s [1]) 
        C: Auger recombination coefficient (e.g. 2.9e-30 cm^6/s [1]) 
    stimulated_decay_rate = g_m * P * confinement_factor/ (hbar * omega * area)
        g_m = a0*(N-N_tr): material power gain coefficient
            There are other possible models. But the ones I found don't seem to be directly applicable
                For quantum well (QW) lasers, g_m = a0*ln(N-N_tr) is a good model.
                For semiconductor optical amplifiers (SOAs), g_m = a0*(N-N_tr) + abar*a0*exp(-N/N0)
            a0: differential gain (e.g. 3.133e-20 m^2 [2])
            N_tr: transparency value of carrier density (e.g. 6.5e23 m^(-3) [2])
        P: optical power in area
        confinement_factor: confinement factor of fundamental mode in 2D waveguide.
            Depends on core thickness, wavelength, n_core, and n_cladding
        area = core thickness * dx: cross-section area of the active region

    Custom definition of n_i (imaginary part of the effective index):
        Power gain model: dP/dz = (g_m * confinement_factor - internal_loss) * P (Equation 2.13 of [4])
        |E|^2 is proportional to P, therefore d|E|/dz = 1/2 * (g_m * confinement_factor - internal_loss) * |E|
        I define n_i such that when sending a uniform input into the 2D waveguide with a uniform n distribution, we have
        |E(z)| = |E(0)| * exp(-n_i * omega / c * z).
        => d|E|/dz = -n_i * omega / c * |E|
        Comapring the two d|E|/dz equations, we get
        n_i = -1/2 * c / omega * (g_m * confinement_factor - internal_loss)
    '''

    # dataclass doesn't actually enforce the datatypes defined below
    dx: u.Quantity
    dz: u.Quantity
    thickness: u.Quantity = 1*u.um
    N0: u.Quantity = 1e11/u.cm**3 # Arbitraty value taken within range of values from [1]. Should be close to actual value. Doesn't matter since it's small.
    A: u.Quantity = 0/u.s # Probably depends on device
    B: u.Quantity = 1.1e-10 * u.cm**3/u.s # From [1]. Should be close to actual according to [3]
    C: u.Quantity = 2.9e-30 * u.cm**6/u.s # From [1]. Depends strongly on x and y of material, need to adjust
    confinement_factor: float = 1 # Need to adjust based on device
    omega: u.Quantity = 2*np.pi * const.c / (1550 * u.nm)
    a0: u.Quantity = 3.133e-16 * u.cm**2 # From [2]
    N_tr: u.Quantity = 6.5e17 / u.cm**3 # From [2]
    dnrdN: u.Quantity = -1e-20 * u.cm**3
    internal_loss: u.Quantity = 0 / u.cm
    
    def __post_init__(self):
        # Custom initialization logic
        self.hbar = const.hbar
        self.e = const.e
        self.c = const.c
        self.current, self.P, self.N = sp.symbols('current P N')
        self.dNdt = self.get_symbolic_dNdt()
        self.steady_state_N_function = sp.lambdify((self.current, self.P), sp.solve(self.dNdt, self.N), 'numpy')
        self.N_value = np.array([])
        self.delta_n = np.array([]) # Definition: delta_n = n - n_real(current = 0, P = 0)

    def __str__(self):
        return pprint.pformat(self.__dict__)
    
    def get_symbolic_dNdt(self):
        # Return a symbolic expression of dN/dt as a function of current, P, N (all need to be their values in SI unit) using sympy (sp)
        values = {}
        attr_names = ['e', 'dx', 'dz', 'thickness', 'N0', 'A', 'B', 'C', 'confinement_factor', 'hbar', 'omega', 'a0', 'N_tr']
        for attr in attr_names:
            value = getattr(self, attr)
            value = to_SI(value)
            values[attr] = value
        e, dx, dz, thickness, N0, A, B, C, confinement_factor, hbar, omega, a0, N_tr = [values[attr] for attr in attr_names]
        current, P, N = self.current, self.P, self.N

        area = dx * thickness
        V = area * dz
        g_m = a0*(N-N_tr)
        
        injection_rate = current/(e * V)
        spontaneous_decay_rate = (N-N0)*(A + B*(N-N0) + C*(N-N0)**2)
        stimulated_decay_rate = g_m * P * confinement_factor/ (hbar * omega * area)
        return injection_rate - spontaneous_decay_rate - stimulated_decay_rate

    def get_steady_state_N(self, current, P):
        # Parameters: injected current and optical power P (in SI unit or Astropy Quantity)
        # Returns: value of steady N (in u.cm**(-3))
        current = np.array(to_SI(current)).astype(np.complex128)
        P = np.array(to_SI(P)).astype(np.complex128)
        result = self.steady_state_N_function(current, P)
        return np.max(np.real(result), axis=0)/(100*u.cm)**3
        #return (result, np.max(np.real(result), axis=0))

    def get_delta_n_r(self, N):
        # Parameter: steady state carrier density N (np array of values in SI unit or Astropy Quantity)
        # Returns: real part of the effective index difference
        return to_SI(N) * to_SI(self.dnrdN)

    def get_n_i(self, N):
        # Parameter: steady state carrier density N (np array of values in SI unit or Astropy Quantity)
        # Returns: imaginary part of the effective n
        N = to_astropy_qty(N, 1/u.m**3)
        g_m = (N - self.N_tr) * self.a0
        return -1/2 * self.c / self.omega * (g_m * self.confinement_factor - self.internal_loss)

    def update(self, current, P, update_delta_n_r = True):
        self.N_value = self.get_steady_state_N(current, P)
        if update_delta_n_r:
            delta_n_r = self.get_delta_n_r(self.N_value)
        else:
            delta_n_r = 0
        n_i = self.get_n_i(self.N_value)
        self.delta_n = np.array(delta_n_r + n_i * 1j).astype(np.complex128)
        return self.delta_n
    
    
########### Utility functions outside of the main classes ##########


def to_SI(value):
    # If valueis an Astropy Quantity, return its value in SI unit
    # Else return itself 
    if isinstance(value, u.Quantity):
        return value.si.value
    else:
        return value

def to_astropy_qty(value, unit):
    # unit must be an Astropy Quantity
    # If value is an Astropy Quantity, return itself
    # Else return value * unit
    if isinstance(value, u.Quantity):
        return value
    else:
        return value * unit

def run_no_voltage_simulation(wg):
    delta_n_store = wg.delta_n
    wg.delta_n = torch.zeros_like(wg.delta_n)
    
    data_V = wg.run_simulation()
    wg.delta_n = delta_n_store
    return data_V

def get_gaussian_kernel(nx, nz, sigmax = 1/3, sigmaz = 1/3):
    # returns array filled with pdf of multivariate gaussian from [-1,1]x[-1,1] 
    # with nx (nz) points in first (second) dimension and standard deviation of sigmax (sigmaz)
    x, z = np.mgrid[-1:1:1j*nx, -1:1:1j*nz] # complex numbers as step size makes this a multidimensional linspace
    pos = np.dstack((x/sigmax, z/sigmaz))
    rv = multivariate_normal([0., 0.])
    weights = rv.pdf(pos)
    weights /= weights.sum()
    return weights

def calc_center_of_gravity(x_axis, beam):
    return torch.sum(torch.from_numpy(x_axis) * beam) / torch.sum(beam)

def create_curved_waveguide(wg, r, input_length, electrode_length, output_length, d_wg, carrier_diffusion_length):
    delta_n = torch.zeros(wg.Nz, wg.Nx)
    xx, zz = np.meshgrid(wg.x_axis, wg.z_axis)

    x0, z0 = r, input_length

    pattern_up = (xx-r)**2 + (zz-z0)**2 > (r-d_wg/2)**2
    pattern_down = (xx-r)**2 + (zz-z0)**2 < (r+d_wg/2)**2

    pattern = pattern_up*pattern_down

    delta_n = torch.from_numpy(pattern.astype(float))*wg.delta_n_val

    delta_n[0:wg.z2ind(input_length), :] = 0.0
    delta_n[wg.z2ind(input_length+electrode_length):wg.z2ind(wg.Lz), :] = 0.0

    delta_n = wg.smoothen_spatial_map(delta_n, carrier_diffusion_length)

    wg.delta_n = delta_n

def single_mode_width(wg):
    return wg.lam0/2/np.sqrt((wg.n+wg.delta_n_val)**2-wg.n**2)
   
from torch.nn import Upsample

def torch_resize(input, scale_factor_1, scale_factor_2):
    # scales a tensor of shape (n1, n2) to (scalscale_factor_1 * n1, scalscale_factor_2 * n2)
    upsample_1 = Upsample(scale_factor=scale_factor_1, mode='nearest') #, align_corners=True)
    upsample_2 = Upsample(scale_factor=scale_factor_2, mode='nearest') #, align_corners=True)
    
    input = input.unsqueeze(0)
    input = upsample_2(input)
    input = input.mT
    input = upsample_1(input)
    input = input.mT.squeeze(0)
    return input

def overlap_intergral(f1, f2):
    return torch.abs(torch.sum(torch.conj_physical(f1) * f2, dim = -1))**2 / torch.norm(f1, dim = -1)**2 / torch.norm(f2, dim = -1)**2

def convolve_with_circle(wg, spatial_map, radius):
    # convolves a spatial map (such as a delta_n) with a circular kernel 
    # of given radius in in um. 
    # The input should have size of [wg.Nx, wg.Nz]
    nx = int(np.round((radius/wg.dx).decompose())/2)*2+1
    nz = int(np.round((radius/wg.dz).decompose())/2)*2+1

    # create circular kernel to average surrounding pixels
    kernel = torch.FloatTensor(
        get_circular_kernel(nz, nx)
    ).unsqueeze(0).unsqueeze(0)
    # pad delta n and convolve it with gaussian kernel
    spatial_map = fft_conv(
        spatial_map.unsqueeze(0).unsqueeze(0), 
        kernel, bias=None, padding = [int(0.5*nz),int(0.5*nx)], 
        padding_mode='reflect')
    return spatial_map.squeeze(0).squeeze(0)

def get_circular_kernel(nx, nz, radius=1):
    # returns a circular kernel of dimension [nx, nz]
    # normalized such that kernel.sum() = 1
    x, z = np.mgrid[-1:1:1j*nx, -1:1:1j*nz] # complex numbers as step size makes this a multidimensional linspace
    pos = np.dstack((x/radius, z/radius))
    dist = np.sqrt((pos**2).sum(axis = -1))
    kernel = (dist<1).astype(float)
    kernel /= kernel.sum()
    return kernel
