from ..shmr.shmr import SHMR
from ..clump_mass_func.clump_mass_func import ClumpMassFunc
from colossus.cosmology import cosmology

class _Unity:
    def __call__(self, cosmo:cosmology, M, z, **kwargs):
        return 1


class _WeightSmoothDM:
    def __init__(self, clump_mass_func:ClumpMassFunc):
        self.clump_mass_func = clump_mass_func

    def __call__(self, cosmo:cosmology, M, z, **kwargs):
        return (1 - self.clump_mass_func.f(M)) * (cosmo.Om(z) - cosmo.Ob(z))/cosmo.Om(z)


class _WeightCentralStar:
    def __init__(self, shmr_central:SHMR):
        self.shmr_central = shmr_central

    def __call__(self, cosmo:cosmology, M, z, **kwargs):
        return self.shmr_central(cosmo, M, z)  


class _WeightGas:
    def __init__(self, shmr:SHMR):
        self.shmr = shmr

    def __call__(self, cosmo:cosmology, M, z, **kwargs):
        return (cosmo.Ob(z)/cosmo.Om(z) -  self.shmr(cosmo, M, z)) 
    
    
class _WeightClumpDM:
    def __call__(self, cosmo:cosmology, M, z, **kwargs):
        return (cosmo.Om(z) - cosmo.Ob(z))/cosmo.Om(z)
    
    
class _WeightClumpStar:
    def __init__(self, clump_mass_func:ClumpMassFunc, shmr:SHMR, shmr_central:SHMR):
        self.shmr = shmr
        self.shmr_central = shmr_central
        self.clump_mass_func = clump_mass_func

    def __call__(self, cosmo:cosmology, m, z, M_parent, **kwargs):
        return (self.shmr(cosmo, m, z) - self.shmr_central(cosmo, m, z)) * self.clump_mass_func.f(M_parent)
    
    
