
from ..concentration import Concentration
from .....config.config import Config


class ConcentrationScaledDistribution(Concentration):
    
    def __init__(self, cfg:Config, concentration_halos: Concentration):
        """
        This model predicts concentration of clumps given a halo concentration.
        :concentration_halos:  concentration model for parent halos
        """
        
        self.cfg = cfg
        self.concentration_halos = concentration_halos


    def _concentration(self, cosmo, m, z):
        return self.cfg.D * self.concentration_halos(cosmo, m, z)

    

