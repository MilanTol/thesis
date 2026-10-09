"""
Base class for computing power spectra. 
To specify dark matter only vs baryon models, 
this must be done through the profiles of halos (see models.py).
"""

from ..halos.profile.profile import Profile
from ..halos.shmr.shmr import SHMR
from ..halos.mass_func.mass_func import MassFunc
from ..halos.clump_mass_func.clump_mass_func import ClumpMassFunc
from ..halos.bias.bias import Bias
from ..config.config import Config
from ..algorithms.integration import MassIntegral
from ..algorithms.grids import tabulate_grid, LogInterpolator2D

import numpy as np
from multiprocessing import Pool, cpu_count
from scipy import interpolate


# The one-halo pieces, in the order they are summed.
COMPONENTS = ("P_1h_ss", "P_1h_sc", "P_1h_self_c", "P_1h_cc")


class MatterPower:

    def __init__(self,
                 cfg: Config,
                 mass_func: MassFunc,
                 smooth_profile: Profile,
                 bias: Bias,
                 clump_mass_func: ClumpMassFunc,
                 clump_profile: Profile,
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
        self.smooth_profile = smooth_profile
        self.bias = bias
        self.clump_mass_func = clump_mass_func
        self.clump_profile = clump_profile
        self.clump_distribution = clump_distribution
        
        self.rho0 = cfg.cosmo.rho_m(z=0) * 1e9 # convert units to M_odot h^2 / Mpc^3
        
        # instantiate mass integral calculator object for both clumps and parent halos:
        self._halo_integrator = MassIntegral(cfg.N_M)
        self._clump_integrator = MassIntegral(cfg.N_m)
        
        # To speed up computation we will interpolate Ic and Jc integrals in the initializer
        self._tabulate_clump_integrals()
        
    def _tabulate_clump_integrals(self) -> None:
        """Tabulate ``I_c`` and ``J_c`` over ``(k, M)`` and build interpolators."""
        cfg = self.cfg
        k_grid = cfg.k_grid()
        M_grid = cfg.M_grid()

        Ic_values, Jc_values = tabulate_grid(
            _clump_integrals_worker, (k_grid, M_grid), extra=(self,),
            n_outputs=2, processes=cfg.processes)

        # Clamped above k_max: both integrals decay to zero on small scales,
        # and extrapolating their linear values would carry the power spectrum
        # negative just past the grid edge.
        self._Ic_interp = LogInterpolator2D(k_grid, M_grid, Ic_values,
                                            clamp_above_x=True)
        self._Jc_interp = LogInterpolator2D(k_grid, M_grid, Jc_values,
                                            clamp_above_x=True)
    
    
    def _compute_Ic(self, k: float, M: float) -> float:
        """Evaluate ``I_c(k, M)``, Giocoli et al. (2010) equation 30.
        The mass-weighted mean clump profile, i.e. how a halo's clumps look in
        Fourier space as a population.
        """
        cfg = self.cfg
        def integrand(m):
            return (
                self.clump_mass_func(m, M)
                * m * self.clump_profile.fourier(cfg.cosmo, k, m, cfg.z, M_parent=M)
            )
        return 1/M * self._clump_integrator(integrand, cfg.m_min, M)
    
    
    def _compute_Jc(self, k: float, M: float) -> float:
        """Evaluate ``J_c(k, M)``, Giocoli et al. (2010) equation 31.
        The clump self-correlation: each clump correlated with itself rather
        than with other clumps.
        """
        cfg = self.cfg
        cfg.activate()

        def integrand(m):
            return (
                self.clump_mass_func(m, M)
                * m * self.clump_profile.fourier(cfg.cosmo, k, m, cfg.z, M_parent=M)
                * m * self.clump_profile.fourier(cfg.cosmo, k, m, cfg.z, M_parent=M)
            )
        return 1/(M*M) * self._clump_integrator(integrand, cfg.m_min, M)
    
    
    def Ic(self, k, M):
        """Interpolated ``I_c(k, M)``."""
        return self._Ic_interp(k, M)

    def Jc(self, k, M):
        """Interpolated ``J_c(k, M)``."""
        return self._Jc_interp(k, M)
    
    
    def M_sm(self, M):
        """Returns the mass in the smooth component given the total halo mass"""
        return (1 - self.clump_mass_func.f(M))*M
    
    
    def P_1h_ss(self, k):
        """Smooth-smooth one-halo term: the smooth halo correlated with itself."""
        def integrand(M):
            return (
                self.mass_func(M, self.cfg.z)
                * self.M_sm(M) * self.smooth_profile.fourier(self.cfg.cosmo, k, self.M_sm(M), self.cfg.z)
                * self.M_sm(M) * self.smooth_profile.fourier(self.cfg.cosmo, k, self.M_sm(M), self.cfg.z)
            )
        return (
            1/(self.rho0*self.rho0) 
            *self._halo_integrator(integrand, self.cfg.M_min, self.cfg.M_max)
        )
        
    
    def P_1h_sc(self, k):
        """Smooth-clump one-halo term: the smooth halo against its own clumps."""
        def integrand(M):
            return (
                2 * self.mass_func(M, self.cfg.z)
                * self.M_sm(M) * self.smooth_profile.fourier(self.cfg.cosmo, k, self.M_sm(M), self.cfg.z)
                * self.clump_distribution.fourier(self.cfg.cosmo, k, M, self.cfg.z)
                * M * self.Ic(k, M)
            )
        return (
            1/(self.rho0*self.rho0) 
            *self._halo_integrator(integrand, self.cfg.M_min, self.cfg.M_max)
        )     


    def P_1h_self_c(self, k):
        """Clump self-correlation one-halo term (Giocoli et al. 2010, eq. 27)."""
        def integrand(M):
            return (
                self.mass_func(M, self.cfg.z)
                * M * M * self.Jc(k, M)
            )
        return (
            1/(self.rho0*self.rho0) 
            *self._halo_integrator(integrand, self.cfg.M_min, self.cfg.M_max)
        )  


    def P_1h_cc(self, k):
        """Clump-clump one-halo term (Giocoli et al. 2010, eq. 28)."""
        def integrand(M):
            return (
                self.mass_func(M, self.cfg.z)
                * self.clump_distribution.fourier(self.cfg.cosmo, k, M, self.cfg.z)
                * self.clump_distribution.fourier(self.cfg.cosmo, k, M, self.cfg.z)
                * M * self.Ic(k, M)
                * M * self.Ic(k, M)
            )
        return (
            1/(self.rho0*self.rho0) 
            *self._halo_integrator(integrand, self.cfg.M_min, self.cfg.M_max)
        )  
    
    
    def P_1h(self ,k):
        """returns the 1-halo power spectrum
        Args:
            k: wavenumber [Mpc/h]^-1
        """
        # sum over strings in components to get all 1h terms.
        return sum(getattr(self, c)(k) for c in COMPONENTS)
    

    def P_lin(self, k):
        """Returns the linear power spectrum"""
        return self.cfg.cosmo.matterPowerSpectrum(k, self.cfg.z)


    def P_tot(self, k):
        """
        Returns the total power on a scale k.
        We approximate the 2h power spectrum with the linear power spectrum, 
        which is sufficiently accurate for the small scales we are interested in in this work.
        """
        return self.P_1h(k) + self.P_lin(k)
    
    
    def components(self, k) -> dict[str, float]:
        """Every component of the spectrum at a single ``k``.

        Returns:
            A dict with the four one-halo pieces, their sum ``P_1h``, the
            two-halo term ``P_2h`` and the total ``P_tot``.
        """
        values = {name: getattr(self, name)(k) for name in COMPONENTS}
        values["P_1h"] = sum(values.values())
        values["P_2h"] = self.P_lin(k)
        values["P_tot"] = values["P_1h"] + values["P_2h"]
        return values
  

# Worker function to compute both Ic and Jc for one (k, M)
def _clump_integrals_worker(args: tuple[MatterPower, float, float]) -> tuple[float, float]:
    model, k, M = args
    return model._compute_Ic(k, M), model._compute_Jc(k, M)
    
