
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import yaml

#Import helper functions
from packages.matter_power_spectrum.helper_functions.model_functions import NFW_functions as NFW
from config.config_cosmology import cosmo

#Parent halo density profile
def smooth_profile(config, k, M):
    """the halo_profile that is being passed to all other functions"""
    
    z = float(config['z'])
    Delta_vir = float(config['Delta_vir'])
    
    return NFW.fourier_NFW(k, M, Delta_vir, z, cosmo)


#Clump mass profile
#Subhalo concentration and subhalo mass function
def concentration_sub(config, m):
    """Subhalo concentration-mass relation from pieri et al."""

    z = float(config['z'])
    d_vir = float(config['d_vir'])

    alpha_R = 0.237
    return np.where(d_vir <= 1, d_vir**-alpha_R * NFW.concentration(m, z), NFW.concentration(m, z))

#Clump NFW profile
def clump_profile(config, k, m):

    z = float(config['z'])

    Delta_vir = config['Delta_vir']
    return NFW.fourier_NFW(k, m, Delta_vir, z, cosmo, c=concentration_sub(config, m))


#Clump distribution
def clump_distribution(config, k, M):
    """the halo_profile that is being passed to all other functions"""
    z = float(config['z'])
    Delta_vir = config['Delta_vir']
    return NFW.fourier_NFW(k, M, Delta_vir, z, cosmo)


#Clump mass function from Giocoli et al. (2010)
def clump_mass_function(config, m, M_parent): #This function is missing c/c_bar, but since I am using fixed c, I believe this is alright?? ............................
    """Clump mass function from Giocoli et al. (2010)"""
    z = float(config['z'])
    A = 9.33e-4 
    alpha = -0.9          
    beta = 12.2715
    return M_parent * (1+z)**0.5 * A * m**alpha * np.exp(-beta * (m / M_parent)**3)


# Fraction of mass in subclumps given clump mass function
def f_sub(config, M_parent):
    """Fraction of mass in subclumps. calculated from clump mass function (giocoli et al. 2010) """
    m_min = float(config['m_min'])
    def integrand(ln_m):
        m = np.exp(ln_m)
        return clump_mass_function(config, m, M_parent) * m
    mass_in_clumps, error = integrate.quad(integrand, np.log(m_min), np.log(M_parent))
    return mass_in_clumps / M_parent

def interpolate_f_sub():
    """
    Interpolates f_sub with current config values.
    """
    global f_sub_interp

    #communicate current task
    print("interpolating f_sub function...")

    with open('/home/milan/Desktop/thesis/code/config/config_matter_power_spectrum.yaml') as cf_file:
        config = yaml.safe_load( cf_file.read() )

    M_min = float(config['M_min'])
    M_max = float(config['M_max'])

    # interpolate f_sub values for faster computation later
    M_values = np.logspace(np.log10(M_min), np.log10(M_max), 100)
    f_sub_values = np.zeros_like(M_values)
    for i, M in enumerate(M_values):
        f_sub_values[i] = f_sub(config, M)
    f_sub_interp = interp1d(M_values, f_sub_values, kind='cubic', fill_value="extrapolate")

