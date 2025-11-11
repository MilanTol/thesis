#imports
import numpy as np

from packages.matter_power_spectrum.helper_functions.sine_cosine_integrals import Si, Ci 


#R_vir and r_c functions (for NFW profile)
def R_vir(M, Delta_vir, z, cosmo):
    """Calculate R_vir given mass M and redshift z."""
    rho_bg = cosmo.rho_m(z=z)*1e9 #converting to Mpc
    return (3 * M / (4 * np.pi * Delta_vir * rho_bg))**(1/3)


# Concentration model
def concentration(M, z): #Model C1
    """concentration-mass relation from Neto et al. (2007) from the aquarius simulation, related to M_200"""
    c0 = 4.67
    beta = -0.11
    M_pivot = 1e14  # in M_sun/h

    return c0 * (M / M_pivot)**beta * (1 + z)**-1


# # Concentration model
# def concentration(M, z): #Model C2
#     """concentration-mass relation from Seljak et al. (2000)"""
#     beta = -0.2
#     M_pivot = 1e14  # in M_sun/h

#     return 9* (1 + z)**-1 * (M / M_pivot)**beta 


# u(k|M): Fourier transform of NFW profile
def fourier_NFW(k, M, Delta_vir, z, cosmo, c=None):
    """Fourier transform of NFW profile normalized by mass."""

    if c is None:
        c = concentration(M, z)
    else:
        pass

    ka = k * R_vir(M, Delta_vir, z, cosmo) / c #In paper by Sheth & Jain referred to as kappa

    def f(c): 
        return 1/(np.log(1 + c) - c / (1 + c))

    temp = f(c)* (np.sin(ka) * (Si((1 + c)*ka) - Si(ka)) 
                - np.sin(c*ka) / ((1 + c)*ka) 
                + np.cos(ka)*(Ci((1 + c)*ka) - Ci(ka)) )
    
    return temp