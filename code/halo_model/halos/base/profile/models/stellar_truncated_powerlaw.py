from ..profile import Profile
from .....config.config import Config
from ...mass_converter import NFWMassConverter
from ...r_vir.models.SO import R_virSO
from ...concentration.concentration import Concentration
from .....algorithms.fourier import fourier

import numpy as np

from scipy.integrate import romb
from scipy.interpolate import RegularGridInterpolator
import numpy as np

from concurrent.futures import ThreadPoolExecutor


class ProfileStellarTruncatedPowerLaw(Profile):
    """
    Instantiates a profile object.
    Exponentially truncated power law for the stellar profile as is done in 
    Siegel et al 2025 (eq. 8)
    """   

    def __init__(self, cfg: Config, c: Concentration):
        self.cfg = cfg
        self.c = c

        print("Computing Fourier transform of stellar profile...")

        k_grid = np.geomspace(cfg.k_min, cfg.k_max, cfg.N_k)
        M_grid = np.geomspace(cfg.M_min, cfg.M_max, 16)

        fourier_grid = np.zeros((len(M_grid), len(k_grid)))

        def compute_row(args):
            i, M = args
            rho_stellar = lambda r: self.real(cfg.cosmo, r, M, cfg.z)
            r_vir = R_virSO(cfg)(cfg.cosmo, M, cfg.z)
            vals = []
            for k in k_grid:
                if k > 1e2:
                    N_r = 128
                else:
                    N_r = 32
                vals.append(fourier(k, rho_stellar, rmin=1e-4, rmax=1e-1, N_r=N_r +1))
            vals = np.array(vals)
            return i, vals / vals[0]

        with ThreadPoolExecutor() as ex:
            for i, vals in ex.map(compute_row, enumerate(M_grid)):
                fourier_grid[i] = vals

        log_fourier = np.log(np.clip(fourier_grid, 1e-5, 1))

        self._interp_2d = RegularGridInterpolator(
            (np.log(M_grid), np.log(k_grid)),
            log_fourier,
            method='cubic',
            bounds_error=False,
            fill_value=None,
        )
        
        # store bounds for to avoid extrapolation
        self._log_M_bounds = (np.log(M_grid[0]),  np.log(M_grid[-1]))
        self._log_k_bounds = (np.log(k_grid[0]),  np.log(k_grid[-1]))
        

    def _fourier(self, cosmo, k, M, z, **kwargs):
        
        # if mass definition is not 200c, convert the mass
        if self.cfg.massdef != "200c":
            mass_converter = NFWMassConverter(self.cfg)
            M = mass_converter(self.cfg.massdef, "200c", M, self.c(cosmo, M, z), z)
            
        k = np.atleast_1d(np.asarray(k, dtype=float))

        log_k = np.clip(np.log(k), *self._log_k_bounds)
        log_M = np.clip(np.log(M), *self._log_M_bounds)

        pts = np.column_stack([
            np.full_like(log_k, log_M),
            log_k
        ])

        result = np.exp(self._interp_2d(pts))
        
        return result if result.shape[0] > 1 else float(result[0])
        
        
        
    # Siegel 2025 uses R200c and R500c definitions (I believe it is critical, due to references to X-ray observations)
    def R200(self, M200, z):
        rho_bg = self.cfg.cosmo.rho_c(z=z)*1e9 #converting to Mpc
        return (3 * M200 / (4 * np.pi * 200 * rho_bg))**(1/3)  
    def R500(self, M500, z):
        rho_bg = self.cfg.cosmo.rho_c(z=z)*1e9 #converting to Mpc
        return (3 * M500 / (4 * np.pi * 500 * rho_bg))**(1/3)            
        
        
    def real(self, cosmo, r, M, z):
        """
        proportional density profile in real space
        """
               
        # if mass definition is not 200c, convert the mass
        if self.cfg.massdef != "200c":
            mass_converter = NFWMassConverter(self.cfg)
            M = mass_converter(self.cfg.massdef, "200c", M, self.c(cosmo, M, z), z)
                
        R_h = self.cfg.r_h * self.R200(M, z)
        
        den = R_h * r**2
        exp = np.exp(-(0.5*r/R_h)**2)
        
        return 1/den * exp
    

        
