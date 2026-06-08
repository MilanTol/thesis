import numpy as np
from scipy.integrate import romb
import mcfit

import warnings

def fourier(
    rho_r:callable, rmin:float, rmax:float, Nr:int, q:float=1.5
) -> np.ndarray:
    
    # setup r_grid
    r_grid = np.geomspace(rmin, rmax, Nr)
    # compute densities along r_grid
    f_r = rho_r(r_grid)
    # don't pass q=0 — use default q=1.5
    # use lowring=True and extrap=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress jax warning
        transform = mcfit.SphericalBessel(r_grid, q=q, lowring=True)
        k, Fk = transform(f_r)
    # correct normalization factor
    Fk = Fk / Fk[0]   
      
    return k, Fk

def get_k_grid(rmin, rmax, Nr):
    """Get the k grid that mcfit will produce for a given r grid."""
    r_grid = np.geomspace(rmin, rmax, Nr)
    dummy = np.ones(Nr)  # profile values don't matter for k grid
    transform = mcfit.SphericalBessel(r_grid, q=1.5, lowring=True)
    k, _ = transform(dummy, extrap=True)
    return k
