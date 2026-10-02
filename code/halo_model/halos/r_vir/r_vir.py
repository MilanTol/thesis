from abc import ABC, abstractmethod

class R_vir(ABC):

    @abstractmethod
    def R_vir(self, cosmo, M, z):
        """Implementation of virial radius"""
        pass        


    def __call__(self, cosmo, M, z):
        """ Returns the virial radius of a halo for input parameters.

        Args:
            M (:obj:`float` or `array`): halo mass.
            z (:obj:`float`): scale factor.

        Returns:
            (:obj:`float` or `array`): Virial Radius 
                :math:`R_{vir}` in units of Mpc/h (comoving).
        """
        return self.R_vir(cosmo=cosmo, M=M, z=0)