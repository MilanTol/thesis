import copy
import numpy as np
import matplotlib.pyplot as plt

from .Pm_computer import Pm_computer
from halo_model.power_spectra.matter_power import MatterPower
from ..config.config import Config


#import ingredient models

from halo_model.halos.base.clump_mass_func.models.giocoli2010_mod import ClumpMassGiocoli2010_mod
from halo_model.halos.base.clump_mass_func.models.giocoli2010 import ClumpMassGiocoli2010
from halo_model.halos.base.clump_mass_func.clump_mass_func import ClumpMassFunc

from halo_model.halos.base.concentration.models.neto2007 import ConcentrationNeto2007
from halo_model.halos.base.concentration.models.scaled_clumps import ConcentrationScaledClumps
from halo_model.halos.base.concentration.models.scaled_distribution import ConcentrationScaledDistribution

from halo_model.halos.base.mass_func.models.tinker2008_mod import MassFuncTinker2008_mod

from halo_model.halos.base.profile.models.nfw import ProfileNFW

from halo_model.halos.base.bias.models.tinker2010 import BiasTinker2010
from halo_model.halos.base.r_vir.models.SO import R_virSO



###################################################################################
# compute Pk
###################################################################################

def compute_Pk(cfg: Config):
    # define model objects
    c_smooth = ConcentrationNeto2007(cfg)
    c_clump = ConcentrationScaledClumps(cfg, c_smooth)
    c_distribution = ConcentrationScaledDistribution(cfg, c_smooth)
    mass_func = MassFuncTinker2008_mod(cfg)
    smooth_profile_dm = ProfileNFW(c_smooth, R_virSO(cfg))
    clump_profile_dm = ProfileNFW(c_clump, R_virSO(cfg))
    bias = BiasTinker2010(cfg)
    clump_distribution = ProfileNFW(c_distribution, R_virSO(cfg))
    

    # computing model power spectra
    cfg_local = copy.deepcopy(cfg)

    # compute CDM power spectrum
    clump_mass_func = ClumpMassGiocoli2010_mod(cfg_local)
    
    Pm = MatterPower(cfg_local, 
                    mass_func=mass_func, 
                    smooth_profile=smooth_profile_dm,
                    bias=bias,
                    clump_mass_func=clump_mass_func, 
                    clump_profile=clump_profile_dm, 
                    clump_distribution=clump_distribution)
        
    # instantiate storage arrays
    k_grid = np.geomspace(cfg.k_min, cfg.k_max, cfg.N_k)
    
    Pm_dict = Pm_computer(Pm, k_grid)
    return Pm_dict


def param_effect_spectrum_DMO(cfg:Config, param:str|list[str], val:float|list[float]):
    cfg_mod = copy.deepcopy(cfg)
    if type(param) is list:
        for i in range(len(param)):
            setattr(cfg_mod, param[i], val[i])
    else:
        setattr(cfg_mod, param, val)
    
    Pm = compute_Pk(cfg)
    Pm_mod = compute_Pk(cfg_mod)
        
    return Pm_mod['P_tot']/Pm['P_tot']
