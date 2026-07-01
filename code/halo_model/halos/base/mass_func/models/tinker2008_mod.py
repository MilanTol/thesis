from ..mass_func import MassFunc
from colossus.lss.mass_function import massFunction
from .....config.config import Config

class MassFuncTinker2008_mod(MassFunc):
    def __init__(self, cfg:Config):
        self.cfg = cfg
        self.massdef = cfg.massdef
        self.delta = int(cfg.massdef[:-1])
        self.ref = cfg.massdef[-1]
        if (self.ref != "c") and (self.ref != "m"):
            raise Exception("reference density not valid for SO model")
                
    def standard(self, M, z): 
        return massFunction(
            M, z=z, mdef = self.massdef, model = 'tinker08', q_out='dndlnM') / M

    def R(self, M):
        return (1 + self.cfg.M0/M)**self.cfg.alpha
    
    def _hmf(self, M, z):
        return self.R(M) * self.standard(M, z)