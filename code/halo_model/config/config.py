import yaml
import os
from colossus.cosmology import cosmology
import copy

class Config:

    def __init__(self, path_to_config=None):
        if path_to_config is None:
            path_to_config = "/home/milan/Desktop/thesis/code/halo_model/config/config.yaml"

        #open the matter power spectrum config file
        with open(path_to_config) as file:
            config = yaml.safe_load(file.read())

        # Matter power spectrum
        self.z          = float(config['z'])
        self.m_min      = float(config['m_min'])
        self.N_m        = int(config['N_m'])
        self.M_min      = float(config['M_min'])
        self.M_max      = float(config['M_max'])
        self.N_M        = int(config['N_M'])
        self.k_min      = float(config['k_min'])
        self.k_max      = float(config['k_max'])
        self.N_k        = int(config['N_k'])
        self.massdef    = config['massdef']
        
        # clump concentration
        self.Q          = float(config['Q'])
        
        # Gas profile
        self.theta_ej   = float(config['theta_ej'])
        self.delta      = float(config['delta'])
        self.gamma      = float(config['gamma'])
        self.mu         = float(config['mu'])
        self.logM_c     = float(config['logM_c'])
        self.logM_gaslimit = float(config['logM_gaslimit'])
        
        # Stellar profile
        self.r_h        = float(config['r_h'])
        
        # SHMR
        self.S          = float(config['S'])

        # Convergence power spectrum
        self.z_min      = float(config['z_min'])
        self.z_max      = float(config['z_max'])
        self.N_z        = int(config['N_z']) # number of points to interpolate in z
        self.l_min      = float(config['l_min'])
        self.l_max      = float(config['l_max'])
        self.N_l        = int(config['N_l'])

        # xi correlation
        self.theta_min  = float(config['theta_min'])
        self.theta_max  = float(config['theta_max'])
        self.N_theta    = int(config['N_theta'])
        
        # Cosmology parameters
        self.cosmo_params = config['cosmo_params']

        cosmology.addCosmology('myCosmo', **self.cosmo_params)
        self.cosmo = cosmology.setCosmology('myCosmo')

        # halo mass function parameters
        self.logM0    = float(config['logM0'])
        self.alpha = float(config['alpha'])

        # clump mass function parameters
        self.logm0   = float(config['logm0'])
        self.beta = float(config['beta'])

        #data storage
        self.Pm_dir     = str(config['Pm_dir'])
        self.Pk_dir     = str(config['Pk_dir'])

    def save(self, path):
        with open(os.path.join(path, 'config.yaml'), 'w') as f:
            yaml.dump(self.to_dict(), f)
    
    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ['cosmo']
        }


def config_modifier(cfg:Config, param:str|list[str], val:float|list[float])->Config:
    cfg_mod = copy.deepcopy(cfg)
    if type(param) is list:
        for i in range(len(param)):
            setattr(cfg_mod, param[i], val[i])
    else:
        setattr(cfg_mod, param, val)
    
    return cfg_mod