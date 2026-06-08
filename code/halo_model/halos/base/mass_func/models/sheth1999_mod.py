from ..mass_func import MassFunc
from colossus.lss.mass_function import massFunction

from scipy.integrate import romb
import numpy as np
from .....config.config import Config


class MassFuncSheth1999_mod(MassFunc):
    def __init__(self, M0, alpha):
        self.M0 = M0
        self.alpha = alpha
        
        # z_linspace = np.linspace(cfg.z_min, cfg.z_max, cfg.N_z)
        # hmf_temp = lambda M, z: self.R(M) * self.standard(M, z)

        # for z in z_linspace:
        #     def M_integrand(lnM):
        #         M = np.exp(lnM)
        #         n = hmf_temp(M, z)
        #         return n * M * M #times M for jacobian dlnM to dM

        # xs = np.linspace(np.log(cfg.M_min), np.log(cfg.M_max), cfg.N_M + 1)
        # dx = xs[1] - xs[0]
        # ys = [M_integrand(lnM) for lnM in xs]

        # return prefactor * romb(ys, dx)

    def standard(self, M, z): 
        """halo mass function from sheth (1999)"""
        return massFunction(M, z=z, model = 'sheth99', q_out='dndlnM') / M

    def R(self, M):
        return (1 + self.M0/M)**self.alpha
    
    def _hmf(self, M, z):
        return self.R(M) * self.standard(M, z)
