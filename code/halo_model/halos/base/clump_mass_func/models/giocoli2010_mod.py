from ..clump_mass_func import ClumpMassFunc
from .....config.config import Config

import numpy as np

class ClumpMassGiocoli2010_mod(ClumpMassFunc):

    def __init__(self, cfg:Config):
        super().__init__(cfg)

    def standard(self, m, M): 
        #This function is missing c/c_bar, 
        #this is only valid for  fixed relation concentration-mass.
        """Clump mass function from Giocoli et al. (2010)"""
        A = 9.33e-4 
        alpha = -0.9          
        beta = 12.2715
        return  (M/m) * (1+self.cfg.z)**0.5 * A * m**alpha * np.exp(-beta * (m / M)**3)

    def R(self, m):
        m0 = 10**self.cfg.logm0
        return (1 + m0/m)**self.cfg.beta
    
    def _cmf(self, m, M):
        return self.R(m) * self.standard(m, M)
    
