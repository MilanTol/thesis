from ..concentration import Concentration
from .....config.config import Config

class ConcentrationNeto2007(Concentration):
    """
    CALIBRATED ON 200m MASS DEFINITION!
    
    concentration-mass relation from Neto et al. (2007) 
    from the aquarius simulation
    """
    
    def __init__(self, cfg:Config):
        if cfg.massdef != "200c":
            raise Exception("Neto2007 requires 200c halo mass definition")
        
        
    def _concentration(self, cosmo, M, z):#Model C1
        c0 = 4.67
        beta = -0.11
        M_pivot = 1e14  # in M_sun/h
        return c0 * (M / M_pivot)**beta * (1 + z)**-1
    

    

