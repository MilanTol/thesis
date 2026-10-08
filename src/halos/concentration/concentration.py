from abc import ABC, abstractmethod

import numpy as np


class Concentration(ABC):
    """
    Base class for concentration models.
    """


    @abstractmethod
    def _concentration(self, cosmo, M, z):
        """Implementation of the c(M) relation."""
        pass


    def __call__(self, cosmo, M, z):
        """Returns the concentration for input parameters.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
            M (:obj:`float` or `array`): halo mass.
            z (:obj:`float`): redshift.

        Returns:
            (:obj:`float` or `array`): concentration.
        """
        M_use = np.atleast_1d(M)
        c = self._concentration(cosmo, M_use, z)
        if np.ndim(M) == 0:
            return c[0]
        return c