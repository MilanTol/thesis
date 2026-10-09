
from ..concentration import Concentration


import numpy as np

class ConcentrationPieri2009_clumps(Concentration):
    
    def __init__(self, concentration_halos: Concentration, d_vir = 1):
        """
        This model predicts concentration of clumps given a halo concentration.
        :concentration_halos:  concentration model for parent halos
        :d_vir: standard distance of clump from parent halo center in units of R_vir 
        """

        self.concentration_halos = concentration_halos
        self.d_vir = d_vir


    def _concentration(self, cosmo, m, z, d_vir = None):
        """clump concentration-mass relation from pieri et al. 2009"""

        if d_vir is None:
            d_vir = self.d_vir

        alpha_R = 0.237
        return np.where(d_vir <= 1, 
                        d_vir**-alpha_R * self.concentration_halos(cosmo, m, z), 
                        self.concentration_halos(cosmo, m, z))

    

