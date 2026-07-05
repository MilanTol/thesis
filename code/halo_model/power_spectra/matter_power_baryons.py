from ..halos.base.profile.profile import Profile, CompositeProfile
from ..halos.base.shmr.shmr import SHMR
from ..halos.base.mass_func.mass_func import MassFunc
from ..halos.base.clump_mass_func.clump_mass_func import ClumpMassFunc
from ..halos.base.bias.bias import Bias
from ..config.config import Config
from colossus.cosmology import cosmology
from ..halos.base.profile.weights import _WeightSmoothDM, _WeightCentralStar, _WeightGas
from ..halos.base.profile.weights import _WeightClumpDM, _WeightClumpStar, _Unity

import numpy as np
from multiprocessing import Pool, cpu_count
from scipy import interpolate
from scipy.integrate import romb


class MatterPowerBaryons:

    def __init__(self,
                 cfg: Config,
                 mass_func: MassFunc,
                 shmr: SHMR,
                 shmr_central: SHMR,
                 smooth_profile_dm: Profile,
                 stellar_profile: Profile,
                 gas_profile: Profile,
                 bias: Bias,
                 clump_mass_func: ClumpMassFunc,
                 clump_profile_dm: Profile,
                 clump_distribution: Profile,
                 ):
        """
        Creates a halo model object
        
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
        self.shmr = shmr
        self.shmr_central = shmr_central
        self.smooth_profile_dm = smooth_profile_dm
        self.stellar_profile = stellar_profile
        self.gas_profile = gas_profile
        self.bias = bias
        self.clump_mass_func = clump_mass_func
        self.clump_profile_dm = clump_profile_dm
        self.clump_distribution = clump_distribution
        
        self.smooth_profile = CompositeProfile(
            smooth_profile_dm, stellar_profile, gas_profile,
            weights=[
                _WeightSmoothDM(clump_mass_func),
                _WeightCentralStar(shmr_central),
                _WeightGas(shmr),
            ],
            # ensure that only the DM profile gets modified because of smooth halo mass
            arg_weights=[
                _WeightSmoothDM(clump_mass_func),
                _Unity(),
                _Unity(),
            ],
            
        )
        self.clump_profile = CompositeProfile(
            clump_profile_dm, stellar_profile,
            weights=[
                _WeightClumpDM(),
                _WeightClumpStar(clump_mass_func, shmr, shmr_central),
            ]    
        )
                
        ######################################################################################
        # To speed up computation we will interpolate Ic and Jc integrals in the initializer
        ######################################################################################
        
        print("interpolating Ic and Jc functions...")

        # Define grids
        lnk_grid = np.log(np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.N_k))
        lnM_grid = np.log(np.logspace(np.log10(cfg.M_min), np.log10(cfg.M_max), cfg.N_M))

        # Allocate arrays
        Ic_vals = np.zeros((len(lnk_grid), len(lnM_grid)))
        Jc_vals = np.zeros((len(lnk_grid), len(lnM_grid)))
    
        args = [(np.log(cfg.m_min), lnM, cfg.N_m, self, np.exp(lnk), np.exp(lnM)) for lnk in lnk_grid for lnM in lnM_grid]

        with Pool(processes=cpu_count()) as pool:
            points = pool.map(compute_point, args)

        # Fill results into arrays
        for (k, M, Ic_val, Jc_val) in points:
            i = np.where(lnk_grid == np.log(k))[0][0]
            j = np.where(lnM_grid == np.log(M))[0][0]
            Ic_vals[i, j] = Ic_val
            Jc_vals[i, j] = Jc_val

        # Create interpolators
        self.Ic_logspace = interpolate.RegularGridInterpolator((lnk_grid, lnM_grid), Ic_vals,
                                                        bounds_error=False, fill_value=None)
        self.Jc_logspace = interpolate.RegularGridInterpolator((lnk_grid, lnM_grid), Jc_vals,
                                                        bounds_error=False, fill_value=None)
    
    def Ic(self, k, M):
        return self.Ic_logspace((np.log(k), np.log(M)))
    
    def Jc(self, k, M):
        return self.Jc_logspace((np.log(k), np.log(M)))
    

    def P_1h_ss(self, k):
        cfg = self.cfg
        
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(lnM):
            M = np.exp(lnM)
            n = self.mass_func(M, cfg.z)

            first_term = n
            # M_sm = (1-self.clump_mass_func.f(M))*M
            # second_term = M_sm**2 * self.smooth_profile.fourier(cfg.cosmo, k, M_sm, cfg.z)**2
            second_term = M**2 * self.smooth_profile.fourier(cfg.cosmo, k, M, cfg.z)**2

            return first_term*second_term*M #times M for jacobian dlnM to dM

        xs = np.linspace(np.log(cfg.M_min), np.log(cfg.M_max), cfg.N_M + 1)
        dx = xs[1] - xs[0]
        ys = [M_integrand(lnM) for lnM in xs]

        return prefactor * romb(ys, dx)
    

    def P_1h_sc(self, k):

        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):
            M = np.exp(ln_M) 
            n = self.mass_func(M, cfg.z)
            first_term = 2 * M  * n

            # M_sm = (1-self.clump_mass_func.f(M))*M
            # second_term = M_sm * self.smooth_profile.fourier(cfg.cosmo, k, M_sm, cfg.z)
            second_term = M * self.smooth_profile.fourier(cfg.cosmo, k, M, cfg.z)

            third_term = self.clump_distribution.fourier(self.cfg.cosmo, k, M, cfg.z) * self.Ic(k, M)

            return first_term * second_term * third_term * M # Jacobian for dlnM to dM conversion
        
        xs = np.linspace(np.log(cfg.M_min), np.log(cfg.M_max), cfg.N_M + 1)
        dx = xs[1] - xs[0]
        ys = [M_integrand(lnM) for lnM in xs]

        return prefactor * romb(ys, dx)
        

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

            first_term = M**2 * n
            return first_term * self.Jc(k, M) * M # Jacobian for dlnM to dM conversion   

        xs = np.linspace(np.log(cfg.M_min), np.log(cfg.M_max), cfg.N_M + 1)
        dx = xs[1] - xs[0]
        ys = [M_integrand(lnM) for lnM in xs]

        return prefactor * romb(ys, dx)


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

            first_term = M**2 * n
            second_term = self.clump_distribution.fourier(cfg.cosmo, k, M, cfg.z)**2 * self.Ic(k, M)**2

            return first_term * second_term * M # Jacobian for dlnM to dM conversion
        
        xs = np.linspace(np.log(cfg.M_min), np.log(cfg.M_max), cfg.N_M + 1)
        dx = xs[1] - xs[0]
        ys = [M_integrand(lnM) for lnM in xs]

        return prefactor * romb(ys, dx)
    
    
    def P_1h(self ,k):
        """returns the 1-halo power spectrum
        Args:
            k: wavenumber [Mpc/h]^-1
        """
        return self.P_1h_ss(k) + self.P_1h_sc(k) + self.P_1h_self_c(k) + self.P_1h_cc(k)
    

    def S_I(self, k):

        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0

        def M_integrand(lnM):

            M = np.exp(lnM)

            n = self.mass_func(M, cfg.z)
            first_term = M * n * self.bias(M, cfg.z)

            second_term = M * self.smooth_profile.fourier(cfg.cosmo, k, M, cfg.z) #1/M is multiplied out by jacobian

            return first_term*second_term #M jacobian is multiplied out by second term 1/M

        xs = np.linspace(np.log(cfg.M_min), np.log(cfg.M_max), cfg.N_M + 1)
        dx = xs[1] - xs[0]
        ys = [M_integrand(lnM) for lnM in xs]

        return prefactor * romb(ys, dx)


    def C_I(self, k):
        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0

        def M_integrand(lnM):

            M = np.exp(lnM)

            n = self.mass_func(M, cfg.z)
            first_term = M * n * self.bias(M, z=cfg.z)

            second_term = self.clump_distribution.fourier(cfg.cosmo, k, M, cfg.z) * self.Ic(k, M)

            return first_term*second_term * M #M jacobian is multiplied out by second term 1/M

        xs = np.linspace(np.log(cfg.M_min), np.log(cfg.M_max), cfg.N_M + 1)
        dx = xs[1] - xs[0]
        ys = [M_integrand(lnM) for lnM in xs]

        return prefactor * romb(ys, dx)


    def P_2h_ss(self, k):
        cfg = self.cfg
        return cfg.cosmo.matterPowerSpectrum(k, cfg.z) * self.S_I(k)**2


    def P_2h_sc(self, k):
        cfg = self.cfg
        return 2*cfg.cosmo.matterPowerSpectrum(k, cfg.z) * self.S_I(k)*self.C_I(k)


    def P_2h_cc(self, k):
        cfg = self.cfg
        return cfg.cosmo.matterPowerSpectrum(k, cfg.z) *self.C_I(k)**2


    def P_2h(self, k):
        cfg = self.cfg # just return the linear matter power spectrum
        return cfg.cosmo.matterPowerSpectrum(k, cfg.z)
        return self.P_2h_ss(k) + self.P_2h_sc(k) + self.P_2h_cc(k)


    def P_tot(self, k):
        return self.P_1h(k) + self.P_2h(k)


def Ic_integrand(lnm, self: MatterPowerBaryons, k, M):
    """I_c integral, eq. (30) in Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
        M_parent: total halo mass [h M_sun]
    """
    m = np.exp(lnm)
    clump_profile_temp = self.clump_profile.fourier(self.cfg.cosmo, k, m, self.cfg.z, M_parent=M)

    return (
        (m/M) 
        * clump_profile_temp 
        * self.clump_mass_func(m, M) 
        * m # Jacobian for dlnm to dm conversion 
    ) 
    

def Ic_analytic(lnm_min, lnm_max, N, self: MatterPowerBaryons, k, M):
    xs = np.linspace(lnm_min, lnm_max, N+1)
    dx = xs[1] - xs[0]
    ys = [Ic_integrand(x, self, k, M) for x in xs]
    return romb(ys, dx, show=False)
    

def Jc_integrand(lnm, self: MatterPowerBaryons, k, M):
    """J_c integral, eq. (31) in Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
        M_parent: total halo mass [h M_sun]
    """

    m = np.exp(lnm)
    return (
        (m/M)**2 
        * self.clump_profile.fourier(self.cfg.cosmo, k, m, self.cfg.z, M_parent=M)**2  
        * self.clump_mass_func(m, M) 
        * m # Jacobian for dlnm to dm conversion
    )   


def Jc_analytic(lnm_min, lnm_max, N, self: MatterPowerBaryons, k, M):
    xs = np.linspace(lnm_min, lnm_max, N+1)
    dx = xs[1] - xs[0]
    ys = [Jc_integrand(x, self, k, M) for x in xs]
    return romb(ys, dx, show=False)
    

# Worker function to compute both Ic and Jc for one (k, M)
def compute_point(args):
    lnm_min, lnm_max, N, Pm, k, M = args
    Ic_val = Ic_analytic(lnm_min, lnm_max, N, Pm, k, M)
    Jc_val = Jc_analytic(lnm_min, lnm_max, N, Pm, k, M)
    return (k, M, Ic_val, Jc_val)