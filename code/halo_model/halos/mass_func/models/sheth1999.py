from ..mass_func import MassFunc
from colossus.lss.mass_function import massFunction

class MassFuncSheth1999(MassFunc):
    
    def _hmf(self, M, z):
        return massFunction(M, z=z, model = 'sheth99', q_out='dndlnM') / M
    
    

class MassFuncSheth1999_mod(MassFunc):
    def __init__(self, M0, alpha):
        self.M0 = M0
        self.alpha = alpha
        
    def standard(self, M, z): 
        """halo mass function from sheth (1999)"""
        return massFunction(M, z=z, model = 'sheth99', q_out='dndlnM') / M

    def R(self, M):
        return (1 + self.M0/M)**self.alpha
    
    def _hmf(self, M, z):
        return self.R(M) * self.standard(M, z)