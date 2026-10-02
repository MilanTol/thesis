from ..concentration import Concentration

class ConcentrationSeljak2000(Concentration):
    
    def _concentration(self, cosmo, M, z):#Model C2
        """concentration-mass relation from Seljak et al. (2000)"""
        beta = -0.2
        M_pivot = 1e14  # in M_sun/h

        return 9* (1 + z)**-1 * (M / M_pivot)**beta 