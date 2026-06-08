from ..profile import Profile
from ...concentration.concentration import Concentration
from ...r_vir.r_vir import R_vir
from scipy.special import sici

import numpy as np

class ProfileNFW(Profile):
    def __init__(self, concentration: Concentration, r_vir: R_vir):
        self.c = concentration
        self.r_vir = r_vir
    

    def f(self, c):
        return 1/(np.log(1 + c) - c / (1 + c))
    
    
    def real(self, cosmo, r, M, z):
        c = self.c(cosmo, M, z)  
        r_s = self.r_vir(cosmo, M, z) / c    
        rho_s = self.f(c) / ( 4*np.pi* r_s**3 )
        
        num = rho_s
        rel_r = r / r_s
        den = rel_r * (1 + rel_r) * (1 + rel_r)
    
        return np.where(r > self.r_vir(cosmo, M, z), 0, num/den)


    def _fourier(self, cosmo, k, M, z, **kwargs):
        c = self.c(cosmo, M, z)

        ka = k * self.r_vir(cosmo, M, z) / c
        
        Si_1cka, Ci_1cka = sici((1 + c)*ka)
        Si_ka, Ci_ka = sici(ka)

        temp = self.f(c)* (np.sin(ka) * (Si_1cka - Si_ka) 
                - np.sin(c*ka) / ((1 + c)*ka) 
                + np.cos(ka)*(Ci_1cka - Ci_ka) )

        return temp
    
