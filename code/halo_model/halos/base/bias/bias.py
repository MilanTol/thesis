from abc import ABC, abstractmethod

class Bias(ABC):
    @abstractmethod
    def _bias(self, M, z):
        pass

    def __call__(self, M, z):
        """
        returns halo bias 
        
        :param M: halo mass
        """
        return self._bias(M, z)
