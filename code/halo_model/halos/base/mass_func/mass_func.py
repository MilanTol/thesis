from abc import ABC, abstractmethod

import numpy as np


class MassFunc(ABC):
    """Base class of halo mass functions.

    .. automethod:: __call__
    """
    
    @abstractmethod
    def _hmf(self, M, z):
        """Implementation of the halo mass function"""
        pass        


    def __call__(self, M, z):
        """ Returns the mass function for input parameters.

        Args:
            M (:obj:`float` or `array`): halo mass.
            z (:obj:`float`): redshift.

        Returns:
            (:obj:`float` or `array`): mass function 
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
        """
        return self._hmf(M=M, z=z)