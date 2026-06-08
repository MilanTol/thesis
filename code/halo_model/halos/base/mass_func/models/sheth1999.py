from ..mass_func import MassFunc
from colossus.lss.mass_function import massFunction

class MassFuncSheth1999(MassFunc):
    
    def _hmf(self, M, z):
        return massFunction(M, z=z, model = 'sheth99', q_out='dndlnM') / M