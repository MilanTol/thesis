
#This code computes the convergence power spectrum from a 3d matter power spectrum.
#For the theory, see ch. 2 and ch. 6 from the review on weak lensing by Bartelmann and Schneider.

import scipy.integrate as integrate

#speed of light
c = 3e5 #km/s

def d(z, cosmo):
    """
    Function returning radial comoving distance.
    Args:
        z: redshift.
        cosmo: cosmology object from colossus.
    """
    return cosmo.comovingDistance(z_min=0, z_max=z, transverse=False) #transverse=False makes sure to calculate the radial comoving distance

def W(z, p, cosmo, z_Hubble):
    """
    Weight function for projecting the 3d matter power spectrum.
    Args:
        z: redshift.
        p: probability distribution of sources over redshift; p(z). If input p is a scalar, it uses a delta function.
        cosmo: cosmology object from colossus.
        z_Hubble: Furthest redshift source.
    """
    if type(p)==float or type(p)==int: #p_z is a delta peak --> all sources at fixed redshift
        return (d(p, cosmo) - d(z, cosmo)) / d(p, cosmo)
    else:
        def integrand(y):
            return p(y) * (d(y, cosmo) - d(z, cosmo)) / d(y, cosmo)
        I, err = integrate.quad(integrand, z, z_Hubble)
        return I
    
def P_convergence(l, P_k_z, p_z, cosmo, z_Hubble):
    """
    Returns convergence power spectrum at value l.
    Args:
        l: angular scale.
        P_k_z: matter matter powerspectrum. Must have only k, z as input; so Pmm = Pmm(k,z).
        p: probability distribution of sources over redshift; p(z). If input p is a scalar, it uses a delta function.
        cosmo: cosmology object from colossus.
        z_Hubble: Furthest redshift source.
    """
    H0 = cosmo.H0
    Om0 = cosmo.Om0
    prefactor = 9 * H0**4 * Om0**2 / (4*c**4) 

    def integrand(z):
        d_temp = d(z, cosmo)
        H = cosmo.Hz(z)
        return c/H * (1+z)**2 * W(z, p_z, cosmo, z_Hubble)**2 * P_k_z(l/d_temp, z) 

    I, err = integrate.quad(integrand, 0, z_Hubble, epsrel=1e-4)

    return prefactor * I
