import copy
import numpy as np
import matplotlib.pyplot as plt

from .Pm_computer import Pm_computer
from halo_model.power_spectra.matter_power_baryons import MatterPowerBaryons

from ..config.config import Config


#import ingredient models

from halo_model.halos.base.clump_mass_func.models.giocoli2010_mod import ClumpMassGiocoli2010_mod
from halo_model.halos.base.clump_mass_func.models.giocoli2010 import ClumpMassGiocoli2010
from halo_model.halos.base.clump_mass_func.clump_mass_func import ClumpMassFunc

from halo_model.halos.base.concentration.models.neto2007 import ConcentrationNeto2007
from halo_model.halos.base.concentration.models.scaled import ConcentrationScaled

from halo_model.halos.base.mass_func.models.tinker2008 import MassFuncTinker2008

from halo_model.halos.base.profile.models.nfw import ProfileNFW
from halo_model.halos.base.profile.models.stellar_truncated_powerlaw import ProfileStellarTruncatedPowerLaw
from halo_model.halos.base.profile.models.gas_cored_powerlaw import ProfileGasCoredPowerLaw

from halo_model.halos.base.bias.models.tinker2010 import BiasTinker2010
from halo_model.halos.base.r_vir.models.SO import R_virSO

from halo_model.halos.base.shmr.models.moster2013 import SHMRMoster2013
from halo_model.halos.base.shmr.models.moster2013_centrals import SHMRMoster2013Centrals
from halo_model.halos.base.shmr.models.Niemiec2022 import SMHRNiemiec2022



###################################################################################
# compute Pk
###################################################################################

def compute_Pk(cfg: Config):
    # define model objects
    c_smooth = ConcentrationNeto2007(cfg)
    c_clump = ConcentrationScaled(cfg, c_smooth)
    mass_func = MassFuncTinker2008(cfg)
    shmr = SHMRMoster2013(cfg, c_smooth)
    shmr_central = SHMRMoster2013Centrals(cfg, c_smooth)
    smooth_profile_dm = ProfileNFW(c_smooth, R_virSO(cfg))
    clump_profile_dm = ProfileNFW(c_clump, R_virSO(cfg))
    bias = BiasTinker2010(cfg)
    clump_distribution = ProfileNFW(c_smooth, R_virSO(cfg))
    

    # computing model power spectra
    cfg_local = copy.deepcopy(cfg)
    gas_profile = ProfileGasCoredPowerLaw(cfg, shmr, c_smooth)
    stellar_profile = ProfileStellarTruncatedPowerLaw(cfg, c_smooth)
    
    # compute CDM power spectrum
    clump_mass_func = ClumpMassGiocoli2010_mod(cfg_local)
    
    Pm = MatterPowerBaryons(cfg_local, 
                    mass_func=mass_func, 
                    shmr=shmr,
                    shmr_central=shmr_central,
                    smooth_profile_dm=smooth_profile_dm,
                    stellar_profile=stellar_profile, 
                    gas_profile=gas_profile,
                    bias=bias,
                    clump_mass_func=clump_mass_func, 
                    clump_profile_dm=clump_profile_dm, 
                    clump_distribution=clump_distribution)
        
    # instantiate storage arrays
    k_grid = np.geomspace(cfg.k_min, cfg.k_max, cfg.N_k)
    
    Pm_dict = Pm_computer(Pm, k_grid)
    return Pm_dict


def param_effect_spectrum(cfg:Config, param:str, val):
    cfg_mod = copy.deepcopy(cfg)
    setattr(cfg_mod, param, val)
    
    Pm = compute_Pk(cfg)
    Pm_mod = compute_Pk(cfg_mod)
        
    return Pm_mod['P_tot']/Pm['P_tot']
