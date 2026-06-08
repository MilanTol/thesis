from .matter_power import MatterPower
from ..halos.base.profile.profile import Profile
from ..halos.base.mass_func.mass_func import MassFunc
from ..halos.base.clump_mass_func.clump_mass_func import ClumpMassFunc
from ..halos.base.bias.bias import Bias
from ..config.config import Config

import numpy as np
from scipy import interpolate
from scipy import integrate
from multiprocessing import Pool, cpu_count

#speed of light
c = 3e5 #km/s

class AngularPower:
    def __init__(self,
                 cfg: Config,
                 mass_func: MassFunc,
                 smooth_profile: Profile,
                 bias: Bias,
                 clump_mass_func: ClumpMassFunc,
                 clump_profile: Profile,
                 clump_distribution: Profile
                 ):
        """       
        :param cfg: Config object
        :type cfg: Config
        :param mass_func: halo mass fucntion
        :type mass_func: MassFunc
        :param bias: bias
        :type bias: Bias
        :param smooth_profile: smooth profile
        :type smooth_profile: Profile
        :param clump_mass_func: Description
        :type clump_mass_func: ClumpMassFunc
        :param clump_profile: Description
        :type clump_profile: Profile
        :param clump_distribution: Description
        :type clump_distribution: Profile
        """
        
        self.cfg = cfg
        self.mass_func = mass_func
        self.smooth_profile = smooth_profile
        self.bias = bias
        self.clump_mass_func = clump_mass_func
        self.clump_profile = clump_profile
        self.clump_distribution = clump_distribution
        
        z_linspace = np.linspace(cfg.z_min, cfg.z_max, cfg.N_z)
        self.Pm_list = []

        #get matter power spectrum at each z value
        for z in z_linspace:
            self.cfg.z = z
            Pm = MatterPower(
                cfg, 
                mass_func=mass_func, 
                smooth_profile=smooth_profile, 
                bias=bias,
                clump_mass_func=clump_mass_func, 
                clump_profile=clump_profile, 
                clump_distribution=clump_distribution
                )
            self.Pm_list.append(Pm)

        print("interpolating power spectra over k and z")

        # Define grids
        lnk_grid = np.log(np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.N_k))

        # Allocate arrays
        P_1h_ss_vals = np.zeros((len(lnk_grid), len(z_linspace)))
        P_1h_sc_vals = np.zeros((len(lnk_grid), len(z_linspace)))
        P_1h_self_c_vals = np.zeros((len(lnk_grid), len(z_linspace)))
        P_1h_cc_vals = np.zeros((len(lnk_grid), len(z_linspace)))
        P_2h_vals = np.zeros((len(lnk_grid), len(z_linspace)))
    
        args = [(np.exp(lnk), Pm) for lnk in lnk_grid for Pm in self.Pm_list]

        with Pool(processes=cpu_count()) as pool:
            points = pool.map(compute_point, args)

        # Fill results into arrays
        for (k, z, P_1h_ss_val, P_1h_sc_val, P_1h_self_c_val, P_1h_cc_val, P_2h_val) in points:
            i = np.where(lnk_grid == np.log(k))[0][0]
            j = np.where(z_linspace == z)[0][0]
            P_1h_ss_vals[i,j] = P_1h_ss_val
            P_1h_sc_vals[i,j] = P_1h_sc_val
            P_1h_self_c_vals[i,j] = P_1h_self_c_val
            P_1h_cc_vals[i,j] = P_1h_cc_val
            P_2h_vals[i,j] = P_2h_val

        # Create interpolators
        self.P_1h_ss_lnk = interpolate.RegularGridInterpolator((lnk_grid, z_linspace), P_1h_ss_vals,
                                                        bounds_error=False, fill_value=None)
        self.P_1h_sc_lnk = interpolate.RegularGridInterpolator((lnk_grid, z_linspace), P_1h_sc_vals,
                                                        bounds_error=False, fill_value=None)
        self.P_1h_self_c_lnk = interpolate.RegularGridInterpolator((lnk_grid, z_linspace), P_1h_self_c_vals,
                                                        bounds_error=False, fill_value=None)
        self.P_1h_cc_lnk = interpolate.RegularGridInterpolator((lnk_grid, z_linspace), P_1h_cc_vals,
                                                        bounds_error=False, fill_value=None)
        self.P_2h_lnk = interpolate.RegularGridInterpolator((lnk_grid, z_linspace), P_2h_vals,
                                                        bounds_error=False, fill_value=None)
        
        
    #matter power spectrum components interpolated over k and z:
    def P_1h_ss(self, k, z):
        return self.P_1h_ss_lnk((np.log(k), z))
    def P_1h_sc(self, k, z):
        return self.P_1h_sc_lnk((np.log(k), z))
    def P_1h_self_c(self, k, z):
        return self.P_1h_self_c_lnk((np.log(k), z))
    def P_1h_cc(self, k, z):
        return self.P_1h_cc_lnk((np.log(k), z))
    def P_1h(self, k, z):
        return self.P_1h_ss(k, z) + self.P_1h_sc(k, z) + self.P_1h_self_c(k,z) + self.P_1h_cc(k,z)
    def P_2h(self, k, z):
        return self.P_2h_lnk((np.log(k), z))
    

    def lensing_kernel(self, z):
        cfg = self.cfg

        H0 = cfg.cosmo.H0 #returns in km/s/Mpc
        Om0 = cfg.cosmo.Om0
        prefactor = 3/2 * Om0 * H0**2 / c**2 
        distance_factor = (cfg.cosmo.comovingDistance(z_max = z)
                            * cfg.cosmo.comovingDistance(z_max = cfg.z_sources - z)
                            / cfg.cosmo.comovingDistance(z_max = cfg.z_sources))
        
        return prefactor * distance_factor * (1+z)


    def P_to_C(self, l, P, w):
        """
        Returns convergence power spectrum at value l using lensing kernel.
        Args:
            l: angular scale.
            P: 3d powerspectrum. Must have only k, z as arguments; so P = P(k,z).
            w: weight function. Must have only z as arguments: so w = w(z)
            cosmo: cosmology object from colossus.
        """
        cfg = self.cfg

        def integrand(z):
            D = cfg.cosmo.comovingDistance(z_max = z)
            H = cfg.cosmo.Hz(z) #returns in km/s/Mpc --> note that c is given in km/s
            return 2*np.pi * c/H * w(z)**2 / D**2 * P(l/D, z)

        I, err = integrate.quad(func=integrand, a=0, b=cfg.z_sources, epsrel=1e-4, limit=200)

        return I
    

    def C_1h_ss(self, l):
        return self.P_to_C(l, self.P_1h_ss, self.lensing_kernel)
    def C_1h_sc(self, l):
        return self.P_to_C(l, self.P_1h_sc, self.lensing_kernel)
    def C_1h_self_c(self, l):
        return self.P_to_C(l, self.P_1h_self_c, self.lensing_kernel)
    def C_1h_cc(self, l):
        return self.P_to_C(l, self.P_1h_cc, self.lensing_kernel)
    def C_1h(self, l):
        return self.P_to_C(l, self.P_1h, self.lensing_kernel)
    def C_2h(self, l):
        return self.P_to_C(l, self.P_2h, self.lensing_kernel)
    def C_tot(self, l):
        return self.C_1h(l) + self.C_2h(l)

# Worker function to compute both Ic and Jc for one (k, M)
def compute_point(args: tuple[float, MatterPower]):
    k, Pm = args
    P_1h_ss_val = Pm.P_1h_ss(k)
    P_1h_sc_val = Pm.P_1h_sc(k)
    P_1h_self_c_val = Pm.P_1h_self_c(k)
    P_1h_cc_val = Pm.P_1h_cc(k)
    P_2h_val = Pm.P_2h(k)
    return (k, Pm.cfg.z, P_1h_ss_val, P_1h_sc_val, P_1h_self_c_val, P_1h_cc_val, P_2h_val)