from ..shmr import SHMR
from .....config.config import Config
from ...mass_converter import NFWMassConverter
from ...concentration.concentration import Concentration

import numpy as np

   
class SHMRMoster2013Centrals(SHMR):
    """
    CALIBRATED ON 200c MASS DEFINITION!
    
    Instantiates a profile object.
    The profile follows delta_function to model the dark matter + stellar component.
    In fourier space this is simply a constant
    """   
    
    def __init__(self, cfg:Config, c:Concentration=None):
        """
        Initialize stellar halo mass relation.
        SMHR CALIBRATED ON 200c MASS DEFINITION!
        
        Args:
            cfg (Config): config file
            c (Concentration, optional): 
                concentration model, should be consistent with massdef from config

        Raises:
            Exception: if no concentration is given while massdef != 200c
        """
        self.cfg = cfg
        self.c = c
        
        if c is None:
            if self.cfg.massdef != "200c":
                raise Exception(
                    "mass definition not consistent with Moster2013 calbiration. \n"
                    "Input a concentration value to convert mass definitions"
                    )

    def _shmr(self, cosmo, M, z): # see moster et al. 2013
        
        # if mass definition is not 200c, convert the mass
        if self.cfg.massdef != "200c":
            mass_converter = NFWMassConverter(self.cfg)
            M = mass_converter(self.cfg.massdef, "200c", M, self.c(self.cfg.cosmo, M, z), z)
            
        # see table 1 moster et al. 2013:
        M10, M11 = 11.590, 1.195
        N10, N11 = 0.0351, -0.0247
        B10, B11 = 1.376, -0.826
        G10, G11 = 0.608, 0.329

        z_fraction = z/(z+1)
        logM1 = M10 + M11*z_fraction
        N = N10 + N11*z_fraction
        beta = B10 + B11*z_fraction
        gamma = G10 + G11*z_fraction

        M1 = 10**logM1
        M1_inv = 1/M1
        f_star = 2*N / ( (M*M1_inv)**(-beta) + (M*M1_inv)**(gamma) )
        return self.cfg.S*f_star
    
