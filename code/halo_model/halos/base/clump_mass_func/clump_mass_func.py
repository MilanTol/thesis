from ..mass_func.mass_func import MassFunc
from ....config.config import Config

from abc import ABC, abstractmethod
from scipy.integrate import romb
from scipy import interpolate
import numpy as np

class ClumpMassFunc(ABC):

    @abstractmethod
    def _cmf(self, m, M):
        """Implementation of the clump mass function"""
        pass
    
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        M_min = float(cfg.M_min)
        M_max = float(cfg.M_max)

        # interpolate f_sub values for faster computation later
        lnM_grid = np.log(np.logspace(np.log10(M_min), np.log10(M_max), cfg.N_M))
        f_values = np.zeros_like(lnM_grid)
        for i, lnM in enumerate(lnM_grid):
            f_values[i] = self._f(np.exp(lnM))
        self.f_lnM = interpolate.interp1d(lnM_grid, f_values, kind='cubic', fill_value="extrapolate")


    def f(self, M:float) -> float:
        """
        returns the fraction of mass contained in clumps in halo of mass M.
        
        :param M: parent halo mass M (total mass, not just smooth component!)
        """
        return self.f_lnM(np.log(M))


    def _f(self, M:float) -> float:
        """
        computes by integration the fraction of mass contained in clumps in halo of mass M
        
        :param M: parent halo mass M (total mass, not just smooth component!)
        """

        def integrand(ln_m):
            m = np.exp(ln_m)
            return m *self._cmf(m, M) * m # additional m factor for lnm to dm conversion
        
        xs = np.linspace(np.log(self.cfg.m_min), np.log(M), self.cfg.N_m + 1)
        dx = xs[1] - xs[0]
        ys = [integrand(lnm) for lnm in xs]

        return romb(ys, dx)/M


    def __call__(self, m, M):
        """ Returns the clump mass function for input parameters.

        Args:
            m (:obj:`float` or `array`): clump mass
            M (:obj:`float` or `array`): parent halo mass (total mass, not just smooth component!).
            z (:obj:`float`): redshift.

        Returns:
            (:obj:`float` or `array`): clump mass function 
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
        """
        return self._cmf(m=m, M=M)