R"""
Parameter sections for every selectable model.
 
Each model gets one frozen dataclass holding its parameters, with no default
values: every value comes from config.yaml. COMPONENTS maps
``component -> {model name -> parameter dataclass}``. To add a model: write its
dataclass, add one line to COMPONENTS, add its section (and selection) to
config.yaml, and register its implementation class in the registry.
 
This module must stay free of heavy imports: Config imports it, and the
profile classes import Config.
"""

from dataclasses import dataclass

###
# gas profiles
###

@dataclass(frozen=True)
class GasSiegel2025:
    """Cored double power law, Siegel et al. 2025 (Table 1)."""
    logM_c: float
    theta_ej: float
    mu: float
    gamma: float
    delta: float
    logM_c: float
    theta_ej: float
    mu: float
    gamma: float
    delta: float
    
    
@dataclass(frozen=True)    
class GasDebackere2022:
    """Beta profile + power law + constant density out to r_h (Debackere et al.).
 
    f_gas,500c(m500c) follows eq. 25-26: a tanh turnover at mass ``10**logM_t``
    with sharpness ``alpha_t``.
    """
    gamma: float        # outer power-law slope, 0 <= gamma <= 3
    logM_t: float      # turnover mass of f_gas,500c [log10 Msun/h]
    alpha_t: float      # sharpness of the turnover
    rc_frac: float      # core radius in units of r500c
    beta_gas: float     # beta-profile slope

        
###
# stellar halo mass relations (SHMR)
###

@dataclass(frozen=True)
class ShmrMoster2013:
    """Stellar halo mass relation adopted from Moster et al. 2013.
    Does not have any parameters to be set.
    """
    

@dataclass(frozen=True)
class ShmrNiemiec2022:
    """Stellar halo mass relation adopted from Niemiec et al. 2022.
    Does not have any parameters to be set.
    """


    
COMPONENTS = {
    "gas_profile": {
        "siegel2025": GasSiegel2025,
        "debackere2022": GasDebackere2022,
    },
    "shmr": {
        "moster2013": ShmrMoster2013,
        "niemiec2022": ShmrNiemiec2022, 
    }
}    
    
    
