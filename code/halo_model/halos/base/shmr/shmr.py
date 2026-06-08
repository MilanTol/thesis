from abc import ABC, abstractmethod
import numpy as np

class SHMR(ABC):
    @abstractmethod
    def _shmr(self, cosmo, M, z):
        pass

    def shmr(self, cosmo, M, z):
        return self._shmr(cosmo, M, z)
    
    def __call__(self, cosmo, M, z):
        return self.shmr(cosmo, M , z)
    


   
