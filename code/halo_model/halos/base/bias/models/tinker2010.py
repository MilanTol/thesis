from ..bias import Bias
from colossus.lss.bias import haloBias
from .....config.config import Config



class BiasTinker2010(Bias):

    def __init__(self, cfg:Config):
        self.massdef = cfg.massdef
        self.delta = int(cfg.massdef[:-1])
        self.ref = cfg.massdef[-1]
        if (self.ref != "c") and (self.ref != "m"):
            raise Exception("reference density not valid for SO model")

    def _bias(self, M, z):
        return haloBias(M, z=z, mdef = self.massdef, model='tinker10')