#I and J integral functions

#imports
import numpy as np
import scipy.integrate as integrate

from . import halo_functions as halo


def I_c(config, k, M_parent):
    """I_c integral, eq. (30) in Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
        M_parent: total halo mass [h M_sun]
    """
        
    m_min = config['m_min']

    def m_integrand(lnm):
        m = np.exp(lnm)
        clump_profile_temp = halo.clump_profile(config, k, m)
        return m/M_parent * clump_profile_temp * 1/m * halo.clump_mass_function(config, m, M_parent) * m # Jacobian for dlnm to dm conversion 

    I, error = integrate.quad(m_integrand, np.log(m_min), np.log(M_parent), epsrel=1e-4, limit=200)

    return I


def J_c(config, k, M_parent):
    """J_c integral, eq. (31) in Giocoli et al..
    Args:
        k: wavenumber [Mpc/h]^-1
        M_parent: total halo mass [h M_sun]
    """

    m_min = config['m_min']

    def m_integrand(lnm):
        m = np.exp(lnm)
        return (m/M_parent)**2 * halo.clump_profile(config, k, m)**2 * 1/m * halo.clump_mass_function(config, m, M_parent) * m # Jacobian for dlnm to dm conversion
    
    J, error = integrate.quad(m_integrand, np.log(m_min), np.log(M_parent), epsrel=1e-4, limit=200)

    return J