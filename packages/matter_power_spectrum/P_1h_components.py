#imports
import numpy as np
from scipy import integrate

#import interpolated Ic and Jc integrals
from .helper_functions import I_and_J_integral_functions_interpolated as I_and_J, halo_functions as halo, lss_functions as lss

#config files
from config.config_cosmology import cosmo
rho_m_0 = cosmo.rho_m(z=0) * 1e9 #convert from 1/kpc^3 to 1/Mpc^3

def P_1h_ss(config, k):
    """Smooth-smooth component for the 1-halo term. See eq. (25) Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
    """
    z = float(config['z'])
    M_min = float(config['M_min'])
    M_max = float(config['M_max'])

    f_sub = halo.f_sub_interp

    def M_integrand(ln_M_parent):

        M_parent = np.exp(ln_M_parent)

        n = lss.halo_mass_function(config, M_parent)
        first_term = (M_parent / rho_m_0)**2 * n

        M_smooth = (1 - f_sub(M_parent)) * M_parent 
        u_smooth = halo.smooth_profile(config, k, M=M_smooth)

        second_term = (M_smooth/M_parent)**2 * u_smooth**2

        return  first_term * second_term * M_parent # Jacobian for dlnM to dM conversion
    
    I, error = integrate.quad(M_integrand, np.log(M_min), np.log(M_max), limit=200, epsrel=1e-4)
    return I


def P_1h_sc(config, k):
    """Smooth-clump component for the 1-halo term. See eq. (26) Giocoli et al.. 
    Args:
        k: wavenumber [Mpc/h]^-1
    """
    
    z = float(config['z'])
    M_min = float(config['M_min'])
    M_max = float(config['M_max'])

    Ic = I_and_J.Ic_interp
    f_sub = halo.f_sub_interp

    def M_integrand(ln_M_parent):

        M_parent = np.exp(ln_M_parent) 

        n = lss.halo_mass_function(config, M_parent)
        first_term = 2 * (M_parent/rho_m_0)**2 * n

        M_smooth = (1 - f_sub(M_parent)) * M_parent  # Smooth mass component
        second_term = M_smooth / M_parent * halo.smooth_profile(config, k, M=M_smooth)

        Ic_val = Ic((k, M_parent))
        third_term = halo.clump_distribution(config, k, M_parent) * Ic_val

        return first_term * second_term * third_term * M_parent # Jacobian for dlnM to dM conversion

    I, error = integrate.quad(M_integrand, np.log(M_min), np.log(M_max), epsrel=1e-4, limit=200)

    return I


def P_1h_self_c(config, k):
    """self-clump component for the 1-halo term. See eq. (27) Giocoli et al.. 
    Args:
        k: wavenumber [Mpc/h]^-1
    """
    z = float(config['z'])
    M_min = float(config['M_min'])
    M_max = float(config['M_max'])

    Jc = I_and_J.Jc_interp

    def M_integrand(ln_M_parent):

        M_parent = np.exp(ln_M_parent)

        n = lss.halo_mass_function(config, M_parent)
        first_term = (M_parent/rho_m_0)**2 * n

        J_c_val = Jc((k, M_parent))
        second_term = J_c_val

        return first_term * second_term * M_parent # Jacobian for dlnM to dM conversion   

    I, error = integrate.quad(M_integrand, np.log(M_min), np.log(M_max), epsrel=1e-4, limit=200)

    return I


def P_1h_cc(config, k):
    """clump-clump component for the 1-halo term. See eq. (28) Giocoli et al.. 
    Args:
        k: wavenumber [Mpc/h]^-1
    """
    z = float(config['z'])
    M_min = float(config['M_min'])
    M_max = float(config['M_max'])

    Ic = I_and_J.Ic_interp

    def M_integrand(ln_M_parent):

        M_parent = np.exp(ln_M_parent)

        n = lss.halo_mass_function(config, M_parent)
        first_term = (M_parent/rho_m_0)**2 * n

        Ic_val = Ic((k, M_parent))
        second_term = halo.clump_distribution(config, k, M_parent)**2 * Ic_val**2

        return first_term * second_term * M_parent # Jacobian for dlnM to dM conversion
    
    I, error = integrate.quad(M_integrand, np.log(M_min), np.log(M_max), epsrel=1e-4, limit=200)

    return I