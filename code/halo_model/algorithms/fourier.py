import numpy as np
from scipy.integrate import romb
import mcfit

# def fourier(k, rho_r:callable, rmin, rmax, N_r=2**7):
#     k = k 
#     def integrand(lnr):
#         r = np.exp(lnr)
#         return 4*np.pi * r*r * np.sinc(k*r / np.pi) * rho_r(r) * r # add factor r for conversion dr -> dr/dlnr dlnr
    
#     lnrs = np.linspace(np.log(rmin), np.log(rmax), N_r)
#     dlnr = lnrs[1] - lnrs[0]
#     ys = [integrand(lnr) for lnr in lnrs]
    
#     return romb(ys, dlnr) 

def fourier(
    rho_r:callable, rmin:float, rmax:float, Nr:int, q:float=1.5
) -> np.ndarray:
    
    # setup r_grid
    r_grid = np.geomspace(rmin, rmax, Nr)
    # compute densities along r_grid
    f_r = rho_r(r_grid)
    # don't pass q=0 — use default q=1.5
    # use lowring=True and extrap=True
    transform = mcfit.SphericalBessel(r_grid, q=q, lowring=True)
    k, Fk = transform(f_r, extrap=True)
    # correct normalization factor
    Fk = Fk / Fk[0] 
    
    print(Fk[0])
    
    return k, Fk


def get_k_grid(rmin, rmax, Nr):
    """Get the k grid that mcfit will produce for a given r grid."""
    r_grid = np.geomspace(rmin, rmax, Nr)
    dummy = np.ones(Nr)  # profile values don't matter for k grid
    transform = mcfit.SphericalBessel(r_grid, q=1.5, lowring=True)
    k, _ = transform(dummy, extrap=True)
    return k
