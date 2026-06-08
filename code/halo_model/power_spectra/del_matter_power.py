from ..halos.base.profile.profile import Profile
from ..halos.base.mass_func.mass_func import MassFunc
from ..halos.base.clump_mass_func.clump_mass_func import ClumpMassFunc
from ..halos.base.bias.bias import Bias
from ..config.config import Config

from scipy import integrate
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy import interpolate

from .matter_power import MatterPower

#Here we rewrite the matterpower class such that it can take in a derivative of a clump mass function


class del_MatterPower(MatterPower):

    def __init__(self,
                 cfg: Config,
                 mass_func: MassFunc,
                 smooth_profile: Profile,
                 bias: Bias,
                 clump_mass_func: ClumpMassFunc,
                 del_clump_mass_func: ClumpMassFunc,
                 clump_profile: Profile,
                 clump_distribution: Profile,
                 epsrel=1e-4
                 ):
        """
        Creates a halo model object that can take in a derivative of a clump mass function
        
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
        :param del_clump_mass_func: derivative of clump mass function wrt to parameter
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
        self.del_clump_mass_func = del_clump_mass_func
        self.clump_profile = clump_profile
        self.clump_distribution = clump_distribution
        self.epsrel = epsrel


        ######################################################################################
        # To speed up computation we will interpolate Ic and Jc integrals in the initializer
        ######################################################################################

        print("interpolating f_sub function...")

        M_min = float(cfg.M_min)
        M_max = float(cfg.M_max)

        # interpolate f_sub values for faster computation later
        lnM_values = np.log(np.logspace(np.log10(M_min), np.log10(M_max), 100))
        f_values = np.zeros_like(lnM_values)
        for i, lnM in enumerate(lnM_values):
            f_values[i] = self.clump_mass_func.f(np.exp(lnM), cfg.z, cfg.m_min)
        self.f_lnM = interpolate.interp1d(lnM_values, f_values, kind='cubic', fill_value="extrapolate")
        
        # interpolate del_f_sub values for faster computation later
        del_f_values = np.zeros_like(lnM_values)
        for i, lnM in enumerate(lnM_values):
            del_f_values[i] = self.del_clump_mass_func.f(np.exp(lnM), cfg.z, cfg.m_min)
        self.del_f_lnM = interpolate.interp1d(lnM_values, del_f_values, kind='cubic', fill_value="extrapolate")

        print("interpolating Ic and Jc functions...")

        # Define grids
        lnk_grid = np.log(np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.N_k))
        lnM_grid = np.log(np.logspace(np.log10(cfg.M_min), np.log10(cfg.M_max), cfg.N_M))

        # Allocate arrays
        Ic_vals = np.zeros((len(lnk_grid), len(lnM_grid)))
        Jc_vals = np.zeros((len(lnk_grid), len(lnM_grid)))
        del_Ic_vals = np.zeros((len(lnk_grid), len(lnM_grid)))
        del_Jc_vals = np.zeros((len(lnk_grid), len(lnM_grid)))

        args = [(self, np.exp(lnk), np.exp(lnM)) for lnk in lnk_grid for lnM in lnM_grid]

        with Pool(processes=cpu_count()) as pool:
            points = pool.map(compute_point, args)

        # Fill results into arrays
        for (k, M, Ic_val, Jc_val, del_Ic_val, del_Jc_val) in points:
            i = np.where(lnk_grid == np.log(k))[0][0]
            j = np.where(lnM_grid == np.log(M))[0][0]
            Ic_vals[i, j] = Ic_val
            Jc_vals[i, j] = Jc_val
            del_Ic_vals[i, j] = del_Ic_val
            del_Jc_vals[i, j] = del_Jc_val

        # Create interpolators
        self.Ic_logspace = interpolate.RegularGridInterpolator((lnk_grid, lnM_grid), Ic_vals,
                                                        bounds_error=False, fill_value=None)
        self.Jc_logspace = interpolate.RegularGridInterpolator((lnk_grid, lnM_grid), Jc_vals,
                                                        bounds_error=False, fill_value=None)
        self.del_Ic_logspace = interpolate.RegularGridInterpolator((lnk_grid, lnM_grid), del_Ic_vals,
                                                        bounds_error=False, fill_value=None)
        self.del_Jc_logspace = interpolate.RegularGridInterpolator((lnk_grid, lnM_grid), del_Jc_vals,
                                                        bounds_error=False, fill_value=None)


    def f(self, M):
        return self.f_lnM(np.log(M))
    def del_f(self, M):
        return self.del_f_lnM(np.log(M))
    
    def Ic(self, k, M):
        return self.Ic_logspace((np.log(k), np.log(M)))
    def Jc(self, k, M):
        return self.Jc_logspace((np.log(k), np.log(M)))
    
    def del_Ic(self, k, M):
        return self.del_Ic_logspace((np.log(k), np.log(M)))
    def del_Jc(self, k, M):
        return self.del_Jc_logspace((np.log(k), np.log(M))) 
   
    def P_1h_ss(self, k):
        cfg = self.cfg
        
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):
            M = np.exp(ln_M)
            M_smooth = (1 - self.f(M)) * M
            del_M_smooth = -self.del_f(M)*M
            n = self.mass_func(M, cfg.z)

            first_term = prefactor * n
            second_term = 2*del_M_smooth*M_smooth * self.smooth_profile.fourier(cfg.cosmo, k, M_smooth, cfg.z)**2

            return first_term*second_term*M #times M for jacobian dlnM to dM

        I, error = integrate.quad(M_integrand, np.log(cfg.M_min), np.log(cfg.M_max), limit=200, epsrel=self.epsrel)
        return I
    

    def P_1h_sc(self, k):
        """
        Docstring for P_1h_sc
        
        :param self: Description
        :param k: Description
        :param z: Description
        :param M_min: Description
        :param M_max: Description
        :param m_min: Description
        """
        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):
            M = np.exp(ln_M) 
            n = self.mass_func(M, cfg.z)
            first_term = 2 * prefactor * M  * n

            M_smooth = (1 - self.f(M)) * M  # Smooth mass component
            del_M_smooth = -self.del_f(M)*M
            second_term = M_smooth * self.smooth_profile.fourier(cfg.cosmo, k, M_smooth, cfg.z)
            del_second_term =  del_M_smooth * self.smooth_profile.fourier(cfg.cosmo, k, M_smooth, cfg.z)

            third_term = self.clump_distribution.fourier(self.cfg.cosmo, k, M, cfg.z) * self.Ic(k, M)
            del_third_term = self.clump_distribution.fourier(self.cfg.cosmo, k, M, cfg.z) * self.del_Ic(k, M)

            return first_term * (del_second_term * third_term + second_term*del_third_term) * M # Jacobian for dlnM to dM conversion

        I, error = integrate.quad(M_integrand, np.log(cfg.M_min), np.log(cfg.M_max), epsrel=self.epsrel, limit=200)

        return I
            

    def P_1h_self_c(self, k):
        """self-clump component for the 1-halo term. See eq. (27) Giocoli et al.. 
        Args:
            k: wavenumber [Mpc/h]^-1
        """
        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):

            M = np.exp(ln_M)
            n = self.mass_func(M, cfg.z)

            first_term = M**2 * prefactor * n
            return first_term * self.del_Jc(k, M) * M # Jacobian for dlnM to dM conversion   

        I, error = integrate.quad(M_integrand, np.log(cfg.M_min), np.log(cfg.M_max), epsrel=self.epsrel, limit=200)

        return I


    def P_1h_cc(self, k):
        """clump-clump component for the 1-halo term. See eq. (28) Giocoli et al.. 
        Args:
            k: wavenumber [Mpc/h]^-1
        """
        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):

            M = np.exp(ln_M)
            n = self.mass_func(M, cfg.z)

            first_term = M**2 * prefactor * n
            M_smooth = (1 - self.f(M))*M
            second_term = (self.clump_distribution.fourier(cfg.cosmo, k, M, cfg.z)**2 
                           * 2*self.del_Ic(k, M) * self.Ic(k, M))

            return first_term * second_term * M # Jacobian for dlnM to dM conversion
        
        I, error = integrate.quad(M_integrand, np.log(cfg.M_min), np.log(cfg.M_max), epsrel=self.epsrel, limit=200)

        return I
    
    
    def P_1h(self ,k):
        """returns the 1-halo power spectrum
        Args:
            k: wavenumber [Mpc/h]^-1
        """
        return self.P_1h_ss(k) + self.P_1h_sc(k) + self.P_1h_self_c(k) + self.P_1h_cc(k)
    


def Ic_analytic(self: MatterPower, k, M):
    """I_c integral, eq. (30) in Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
        M_parent: total halo mass [h M_sun]
    """
    
    def m_integrand(lnm):
        m = np.exp(lnm)
        clump_profile_temp = self.clump_profile.fourier(self.cfg.cosmo, k, m, self.cfg.z)

        return (1/M 
                * clump_profile_temp 
                * self.clump_mass_func(m, M, self.cfg.z) 
                * m) # Jacobian for dlnm to dm conversion 

    I, error = integrate.quad(m_integrand, np.log(self.cfg.m_min), np.log(M), epsrel=self.epsrel, limit=200)

    return I


def Jc_analytic(self: MatterPower, k, M):
    """J_c integral, eq. (31) in Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
        M_parent: total halo mass [h M_sun]
    """

    def m_integrand(lnm):
        m = np.exp(lnm)
        return (m*(1/M)**2 
                * self.clump_profile.fourier(self.cfg.cosmo, k, m, self.cfg.z)**2  
                * self.clump_mass_func(m, M, self.cfg.z) 
                * m)   # Jacobian for dlnm to dm conversion
    
    J, error = integrate.quad(m_integrand, np.log(self.cfg.m_min), np.log(M), epsrel=self.epsrel, limit=200)

    return J


def del_Ic_analytic(self: del_MatterPower, k, M):
    """I_c integral, eq. (30) in Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
        M_parent: total halo mass [h M_sun]
    """
    
    def m_integrand(lnm):
        m = np.exp(lnm)
        clump_profile_temp = self.clump_profile.fourier(self.cfg.cosmo, k, m, self.cfg.z)

        return (1/M 
                * clump_profile_temp 
                * self.del_clump_mass_func(m, M, self.cfg.z) 
                * m) # Jacobian for dlnm to dm conversion 

    I, error = integrate.quad(m_integrand, np.log(self.cfg.m_min), np.log(M), epsrel=self.epsrel, limit=200)

    return I


def del_Jc_analytic(self: del_MatterPower, k, M):
    """J_c integral, eq. (31) in Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
        M_parent: total halo mass [h M_sun]
    """

    def m_integrand(lnm):
        m = np.exp(lnm)
        return (m*(1/M)**2 
                * self.clump_profile.fourier(self.cfg.cosmo, k, m, self.cfg.z)**2  
                * self.del_clump_mass_func(m, M, self.cfg.z) 
                * m)   # Jacobian for dlnm to dm conversion
    
    J, error = integrate.quad(m_integrand, np.log(self.cfg.m_min), np.log(M), epsrel=self.epsrel, limit=200)

    return J


# Worker function to compute both Ic and Jc for one (k, M)
def compute_point(args):
    Pm, k, M = args
    Ic_val = Ic_analytic(Pm, k, M)
    Jc_val = Jc_analytic(Pm, k, M)
    del_Ic_val = del_Ic_analytic(Pm, k, M)
    del_Jc_val = del_Jc_analytic(Pm, k, M)
    return (k, M, Ic_val, Jc_val, del_Ic_val, del_Jc_val)

