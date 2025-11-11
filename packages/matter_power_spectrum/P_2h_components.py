#imports
import numpy as np
from scipy import integrate

#import interpolated Ic and Jc integrals
from packages.matter_power_spectrum.helper_functions import I_and_J_integral_functions_interpolated as I_and_J, halo_functions as halo, lss_functions as lss

#config files
from config.config_cosmology import cosmo
rho_m_0 = cosmo.rho_m(z=0) * 1e9 #convert from 1/kpc^3 to 1/Mpc^3


def S_I(config, k):

    z = float(config['z'])
    M_min = float(config['M_min'])
    M_max = float(config['M_max'])

    f_sub = halo.f_sub_interp
    
    def M_integrand(lnM_parent):

        M_parent = np.exp(lnM_parent)

        n = lss.halo_mass_function(config, M_parent)
        first_term = M_parent/rho_m_0 * n * lss.halo_bias(config, M_parent)

        M_smooth = (1 - f_sub(M_parent)) * M_parent
        u_smooth = halo.smooth_profile(config, k, M=M_smooth)
        second_term = M_smooth/M_parent * u_smooth

        return first_term*second_term * M_parent

    S, error = integrate.quad(M_integrand, np.log(M_min), np.log(M_max), epsrel=1e-4, limit=200)
    return S


def C_I(config, k):

    z = float(config['z'])
    M_min = float(config['M_min'])
    M_max = float(config['M_max'])

    Ic = I_and_J.Ic_interp
    f_sub = halo.f_sub_interp
    
    def M_integrand(lnM_parent):
        M_parent = np.exp(lnM_parent)

        n = lss.halo_mass_function(config, M_parent)
        first_term = M_parent/rho_m_0 * n * lss.halo_bias(config, M_parent)

        M_smooth = (1 - f_sub(M_parent)) * M_parent
        U = halo.clump_distribution(config, k, M=M_smooth)
        second_term = Ic((k, M_parent)) * U

        return first_term*second_term * M_parent

    C, error = integrate.quad(M_integrand, np.log(M_min), np.log(M_max), epsrel=1e-4, limit=200)
    return C


#Compute 2h smooth smooth values
def P_2h_ss(config, k):
    z = float(config['z'])
    return cosmo.matterPowerSpectrum(k, z) * S_I(config, k)**2


#Compute 2h smooth clump values
def P_2h_sc(config, k):
    z = float(config['z'])
    return 2*cosmo.matterPowerSpectrum(k, z) * S_I(config, k)*C_I(config, k)


#Compute 2h clump clump values
def P_2h_cc(config, k):
    z = float(config['z'])
    return cosmo.matterPowerSpectrum(k, z) *C_I(config, k)**2
