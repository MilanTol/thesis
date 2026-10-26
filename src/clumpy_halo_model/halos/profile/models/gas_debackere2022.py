from ..profile import Profile
from ...shmr.shmr import SHMR
from ....config.config import Config
from ....algorithms.mass_converter import NFWMassConverter
from ...concentration.concentration import Concentration

from ....algorithms.fourier import fourier, get_k_grid

from scipy.interpolate import RegularGridInterpolator
from scipy.special import hyp2f1
import numpy as np

from concurrent.futures import ThreadPoolExecutor


class ProfileGasDebackere2022(Profile):
    """
    Hot gas profile as in Debackere et al. (eq. 14-15):
      - beta profile for r < r500c
      - power law with slope -gamma for r500c <= r < r200m
      - constant density for r200m <= r < r_h
      - zero for r >= r_h
    rho_0 is set by the observed gas fraction at r500c (eq. 15).
    r_h is the radius where the halo reaches the cosmic baryon fraction
    (gas + stars = f_b * M_h), which constrains gamma.
    Returns the physical density (Msun / Mpc^3), not a proportional profile.
    """

    def __init__(self, cfg: Config, shmr: SHMR, c: Concentration):
        self.cfg = cfg
        self._shmr = shmr
        self.c = c
        self.p = cfg.p("gas_profile")

        rmin, rmax, Nr = 1e-4, 1e3, 512
        self.M_gaslimit = 10**(cfg.logM_gaslimit)
        M_grid = np.geomspace(self.M_gaslimit, cfg.M_max, 32)
        k_grid = get_k_grid(rmin, rmax, Nr)
        fourier_grid = np.zeros((len(M_grid), len(k_grid)))

        def compute_row(args):
            i, M = args
            rho_r = lambda r: self.real(cfg.cosmo, r, M, cfg.z)
            k, Fk = fourier(rho_r, rmin, rmax, Nr)
            return i, Fk

        with ThreadPoolExecutor() as ex:
            for i, vals in ex.map(compute_row, enumerate(M_grid)):
                fourier_grid[i] = vals

        self._interp_2d = RegularGridInterpolator(
            (np.log(M_grid), np.log(k_grid)),
            fourier_grid,
            method='cubic',
            bounds_error=False,
            fill_value=None,
        )

    def _fourier(self, cosmo, k, M, z, **kwargs):
        # M_gaslimit is defined in cfg.massdef here, so no conversion is needed
        M = np.maximum(M, self.M_gaslimit)
        k = np.atleast_1d(np.asarray(k, dtype=float))

        log_k = np.log(k)
        log_M = np.log(M)

        pts = np.column_stack([
            np.full_like(log_k, log_M),
            log_k
        ])

        result = self._interp_2d(pts)
        return result if result.shape[0] > 1 else float(result[0])

    # ---------- helpers ----------
    def _rho_ref(self, z, kind):
        # reference density in Msun/Mpc^3 (same *1e9 convention as the Siegel file)
        if kind == "c":
            return self.cfg.cosmo.rho_c(z=z) * 1e9
        return self.cfg.cosmo.rho_m(z=z) * 1e9  

    def R_delta(self, M, delta, kind, z):
        return (3 * M / (4 * np.pi * delta * self._rho_ref(z, kind)))**(1/3)

    def _convert(self, cosmo, M, z, target):
        if self.cfg.massdef == target:
            return M
        mass_converter = NFWMassConverter(self.cfg)
        return mass_converter(self.cfg.massdef, target, M, self.c(cosmo, M, z), z)

    # ---------- observational fit parameters (Debackere Sec. 3) ----------
    def f_gas_500(self, M500, M_h, z):
        # eq. 25: sigmoid-like fit, -> f_b for M -> inf, -> 0 for M -> 0
        f_b = self.cfg.cosmo.Ob(z) / self.cfg.cosmo.Om(z)  # Omega_b / Omega_m
        m_t = 10**self.p.logM_t  # turnover mass
        alpha = self.p.alpha_t  # sharpness of the turnover
        f_sig = 0.5 * f_b * (1 + np.tanh(np.log10(M500 / m_t) / alpha))
        # eq. 26: f_gas <= f_b - f_star,500c
        # ASSUMPTION: all stellar mass of the halo sits inside r500c
        # this is maybe not correct when clumps also contain stars?
        f_star = self._shmr.shmr(self.cfg.cosmo, M_h, z)  
        return np.minimum(f_sig, np.maximum(f_b - f_star, 0.0))

    # ---------- normalisation (eq. 15) ----------
    def rho_0(self, M500, r500, rc, beta, M_h, z):
        fgas = self.f_gas_500(M500, M_h, z)
        F = hyp2f1(1.5, 1.5 * beta, 2.5, -(r500 / rc)**2)
        return 500 * self._rho_ref(z, "c") * fgas / F
    
    # ---------- gas mass enclosed in power-law / constant parts ----------
    @staticmethod
    def _m_powerlaw(rho500, r500, gamma, r_lo, r_hi):
        # int 4 pi r^2 rho500 (r/r500)^-gamma dr from r_lo to r_hi
        if np.isclose(gamma, 3.0):
            return 4 * np.pi * rho500 * r500**3 * np.log(r_hi / r_lo)
        return 4 * np.pi * rho500 * r500**gamma * (r_hi**(3 - gamma) - r_lo**(3 - gamma)) / (3 - gamma)


    def r_halo(self, M_h, M500, r500, r200m, rho500, gamma, z):
        """
        Radius where gas + stars = f_b * M_h.
        """
        f_b = self.cfg.cosmo.Ob(z) / self.cfg.cosmo.Om(z)
        M_star = M_h * self._shmr.shmr(self.cfg.cosmo, M_h, z)  
        M_gas_target = f_b * M_h - M_star
        m500 = self.f_gas_500(M500, M_h, z) * M500
        m200m = m500 + self._m_powerlaw(rho500, r500, gamma, r500, r200m)
        rho_out = rho500 * (r200m / r500)**(-gamma)

        if M_gas_target >= m200m:
            # constant density region extends past r200m
            return (r200m**3 + 3 * (M_gas_target - m200m) / (4 * np.pi * rho_out))**(1/3)

        # r_h lies inside r200m: truncate the power law
        remaining = M_gas_target - m500
        if remaining <= 0:
            return r500  # edge case: baryon budget below the r500 gas mass, clip
        if np.isclose(gamma, 3.0):
            return r500 * np.exp(remaining / (4 * np.pi * rho500 * r500**3))
        arg = r500**(3 - gamma) + remaining * (3 - gamma) / (4 * np.pi * rho500 * r500**gamma)
        return arg**(1 / (3 - gamma))


    def real(self, cosmo, r, M, z):
        """
        hot gas density in real space (Msun / Mpc^3), M in cfg.massdef
        """
        r = np.atleast_1d(np.asarray(r, dtype=float))

        M500 = self._convert(cosmo, M, z, "500c")
        M200m = self._convert(cosmo, M, z, "200m")
        r500 = self.R_delta(M500, 500, "c", z)
        r200m = self.R_delta(M200m, 200, "m", z)
        
        rc = self.p.rc_frac*r500
        rho0 = self.rho_0(M500, r500, rc, self.p.beta, M, z)
        rho500 = rho0 * (1 + (r500 / rc)**2)**(-1.5 * self.p.beta)
        
        gamma_requested = self.p.gamma
        gamma_min = self.gamma_min(
            M, M500, r500, r200m, rho500, z
        )
        gamma = max(gamma_requested, gamma_min)
        
        r_h = self.r_halo(M, M500, r500, r200m, rho500, gamma, z)

        inner = rho0 * (1 + (r / rc)**2)**(-1.5 * self.p.beta)
        middle = rho500 * (r / r500)**(-gamma)
        outer = rho500 * (r200m / r500)**(-gamma)

        return np.where(
            r < r500, inner,
            np.where(r < min(r200m, r_h), middle,
                     np.where(r < r_h, outer, 0.0))
        ) 
    
    # helper
    def gamma_min(
        self, M_h, M500, r500, r200m, rho500, z
    ):
        """
        Minimum power-law slope required for the gas mass
        within r200m not to exceed the available baryon budget.

        Assumes the halo mass M_h and stellar mass are defined
        consistently with the baryon-budget prescription.
        """
        f_b = self.cfg.cosmo.Ob(z) / self.cfg.cosmo.Om(z)

        M_star = M_h*self._shmr.shmr(self.cfg.cosmo, M_h, z)
        M_gas_target = f_b * M_h - M_star

        M_gas_500 = self.f_gas_500(
            M500, M_h, z
        ) * M500

        tol = 1e-10 * max(abs(M_gas_target), abs(M_gas_500), 1.0)

        # The available gas budget must accommodate the gas
        # already enclosed within r500.
        if M_gas_target < M_gas_500 - tol:
            raise ValueError(
                "The gas mass inside r500 already exceeds the "
                "available baryon budget. Check the mass "
                "definitions, SHMR, and gas-fraction normalization."
            )

        def gas_mass_200(gamma):
            return M_gas_500 + self._m_powerlaw(
                rho500, r500, gamma, r500, r200m
            )

        # If gamma=0 is allowed, no additional constraint is needed.
        if gas_mass_200(0.0) <= M_gas_target:
            return 0.0

        # If even gamma=3 fails, the problem cannot be solved
        # by changing gamma alone.
        M_gas_200_gamma3 = gas_mass_200(3.0)

        # if M_gas_200_gamma3 > M_gas_target + tol:
        #     print("\n--- gamma_min diagnostic ---")
        #     print(f"M_h                 = {M_h:.6e}")
        #     print(f"M500                = {M500:.6e}")
        #     print(f"r500                = {r500:.6e}")
        #     print(f"r200m               = {r200m:.6e}")
        #     print(f"rho500              = {rho500:.6e}")
        #     print(f"f_b                 = {f_b:.6e}")
        #     print(f"M_star              = {M_star:.6e}")
        #     print(f"M_gas_target        = {M_gas_target:.6e}")
        #     print(f"M_gas_500            = {M_gas_500:.6e}")
        #     print(f"M_gas_200_gamma0     = {gas_mass_200(0.0):.6e}")
        #     print(f"M_gas_200_gamma3     = {M_gas_200_gamma3:.6e}")
        #     print(f"gas_fraction_500     = {M_gas_500 / M500:.6e}")
        #     print(
        #         f"required_gamma3_ratio = "
        #         f"{M_gas_200_gamma3 / M_gas_target:.6e}"
        #         if M_gas_target > 0 else
        #         "M_gas_target <= 0"
        #     )
        #     print("----------------------------\n")

        #     raise ValueError(
        #         "The gas mass exceeds the available baryon budget "
        #         "even for gamma=3."
        #     )

        # Bisection: find the smallest gamma satisfying the constraint.
        lo, hi = 0.0, 3.0

        for _ in range(60):
            mid = 0.5 * (lo + hi)

            if gas_mass_200(mid) > M_gas_target:
                lo = mid
            else:
                hi = mid

        return hi    