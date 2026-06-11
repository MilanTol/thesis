from ..profile import Profile
from ...shmr.shmr import SHMR
from .....config.config import Config
from ...mass_converter import NFWMassConverter
from ...r_vir.models.SO import R_virSO
from ...concentration.concentration import Concentration

from .....algorithms.fourier import fourier, get_k_grid

from scipy.integrate import romb
from scipy.interpolate import RegularGridInterpolator
import numpy as np

  
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
import numpy as np


class ProfileGasCoredPowerLaw(Profile):
    """
    Instantiates a profile object.
    Cored double powerlaw for the gas profile as is done in 
    Siegel et al 2025 (eq. 6)
    """   

    def __init__(self, cfg: Config, shmr: SHMR, c: Concentration):
        self.cfg = cfg
        self._shmr = shmr
        self.c = c

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
        
        # if mass definition is not 200c, convert the mass
        if self.cfg.massdef != "200c":
            mass_converter = NFWMassConverter(self.cfg)
            M = mass_converter(self.cfg.massdef, "200c", M, self.c(cosmo, M, z), z)
        
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
    
    
    # Siegel 2025 uses R200c and R500c definitions (I believe it is critical, due to references to X-ray observations)
    def R200(self, M200, z):
        rho_bg = self.cfg.cosmo.rho_c(z=z)*1e9 #converting to Mpc
        return (3 * M200 / (4 * np.pi * 200 * rho_bg))**(1/3)            
        
        
    def beta(self, M200):
        # See Table 1 Siegel et al. 2025
        M_c = 10**self.cfg.logM_c
        mu = self.cfg.mu
        m = M200 / M_c
        return 3*m**mu / (1 + m**mu)    
        
        
    def real(self, cosmo, r, M, z):
        """
        proportional density profile in real space
        """
        # if mass definition is not 200c, convert the mass
        if self.cfg.massdef != "200c":
            mass_converter = NFWMassConverter(self.cfg)
            M = mass_converter(self.cfg.massdef, "200c", M, self.c(cosmo, M, z), z)
                    
        theta_ej = self.cfg.theta_ej # see Table 1 Siegel et al. 2025
        delta = self.cfg.delta
        gamma = self.cfg.gamma
                       
        r_core = 0.1*self.R200(M, z)
        r_ej = theta_ej * self.R200(M, z)
        
        den = (1 + (r/r_core))**self.beta(M) * (1 + (r/r_ej)**gamma)**( (delta - self.beta(M)) / gamma )
        
        return 1/den
    

   
  
