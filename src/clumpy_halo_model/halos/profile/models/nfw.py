from ..profile import Profile
from ...concentration.concentration import Concentration
from ...r_vir.r_vir import R_vir
from scipy.special import sici

from ..profile import Profile
from ....config.config import Config

from ....algorithms.fourier import fourier, get_k_grid

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ThreadPoolExecutor


class ProfileNFW(Profile):
    def __init__(self, cfg: Config, c: Concentration, r_vir: R_vir):
        self.c = c
        self.r_vir = r_vir

        # rmin, rmax, Nr = 1e-4, 5e3, 512
        # M_grid = np.geomspace(cfg.M_min, cfg.M_max, cfg.N_M)
        # k_grid = get_k_grid(rmin, rmax, Nr)
        # fourier_grid = np.zeros((len(M_grid), len(k_grid)))

        # def compute_row(args):
        #     i, M = args
        #     rho_r = lambda r: self.real(cfg.cosmo, r, M, cfg.z)
        #     k, Fk = fourier(rho_r, rmin, rmax, Nr)
        #     return i, Fk

        # with ThreadPoolExecutor() as ex:
        #     for i, vals in ex.map(compute_row, enumerate(M_grid)):
        #         fourier_grid[i] = vals

        # self._interp_2d = RegularGridInterpolator(
        #     (np.log(M_grid), np.log(k_grid)),
        #     fourier_grid,
        #     method='cubic',
        #     bounds_error=False,
        #     fill_value=None,
        # )

    def f(self, c):
        return 1/(np.log(1 + c) - c / (1 + c))
    
    
    def real(self, cosmo, r, M, z):
        c = self.c(cosmo, M, z)  
        r_s = self.r_vir(cosmo, M, z) / c    
        rho_s = self.f(c) / ( 4*np.pi* r_s**3 )
        
        num = rho_s
        rel_r = r / r_s
        den = rel_r * (1 + rel_r) * (1 + rel_r)
    
        # return num/den 
        return np.where(r > self.r_vir(cosmo, M, z), 0, num/den)

    def _fourier(self, cosmo, k, M, z, **kwargs):
        return self.fourier_analytic(cosmo, k, M, z, **kwargs)
        # # M_gaslimit is defined in cfg.massdef here, so no conversion is needed
        # k = np.atleast_1d(np.asarray(k, dtype=float))

        # log_k = np.log(k)
        # log_M = np.log(M)

        # pts = np.column_stack([
        #     np.full_like(log_k, log_M),
        #     log_k
        # ])
        
        # result = self._interp_2d(pts)
        
        # return result if result.shape[0] > 1 else float(result[0])
    

    def fourier_analytic(self, cosmo, k, M, z, **kwargs):
        c = self.c(cosmo, M, z)

        ka = k * self.r_vir(cosmo, M, z) / c
        
        Si_1cka, Ci_1cka = sici((1 + c)*ka)
        Si_ka, Ci_ka = sici(ka)

        temp = self.f(c)* (np.sin(ka) * (Si_1cka - Si_ka) 
                - np.sin(c*ka) / ((1 + c)*ka) 
                + np.cos(ka)*(Ci_1cka - Ci_ka) )

        return temp