# Imports
from scipy import interpolate
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import yaml

# import I_c and J_c functions
from .I_and_J_integral_functions import I_c
from .I_and_J_integral_functions import J_c


def update_values():
    """
    Updates global variables within I_and_J_integral_functions_interpolated.py to config values
    """
    global k_min
    global k_max
    global M_min
    global M_max
    global compute_point

    with open('/home/milan/Desktop/thesis/code/config/config_matter_power_spectrum.yaml') as cf_file:
        config = yaml.safe_load( cf_file.read() )

    k_min = float(config['k_min'])
    k_max = float(config['k_max'])
    M_min = float(config['M_min'])
    M_max = float(config['M_max'])

    # Worker function to compute both Ic and Jc for one (k, M)
    def compute_point(k, M):
        Ic_val = I_c(config, k, M)
        Jc_val = J_c(config, k, M)
        return (k, M, Ic_val, Jc_val)


def interpolate_I_and_J():
    """
    Interpolates I and J functions as stores them as Ic_interp, Jc_interp.
    """
    global Ic_interp
    global Jc_interp

    update_values() #updates all necessary values based on current config file

    #communicate current task
    print("interpolating Ic and Jc functions...")

    # Define grids
    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), 30)  
    M_grid = np.logspace(np.log10(M_min), np.log10(M_max), 60)   

    # Allocate arrays
    Ic_vals = np.zeros((len(k_grid), len(M_grid)))
    Jc_vals = np.zeros((len(k_grid), len(M_grid)))

    #Run in parallel
    delayed_calls = [delayed(compute_point)(k, M) for k in k_grid for M in M_grid]
    output = Parallel(n_jobs=-1, backend='threading')(delayed_calls) #Use threading when __name__ neq __main__
    output = np.array(output)

    # Fill results into arrays
    for (k, M, Ic_val, Jc_val) in output:
        i = np.where(k_grid == k)[0][0]
        j = np.where(M_grid == M)[0][0]
        Ic_vals[i, j] = Ic_val
        Jc_vals[i, j] = Jc_val

    # Create interpolators
    Ic_interp = interpolate.RegularGridInterpolator((k_grid, M_grid), Ic_vals,
                                                    bounds_error=False, fill_value=None)
    print("Ic function interpolated")
    Jc_interp = interpolate.RegularGridInterpolator((k_grid, M_grid), Jc_vals,
                                                    bounds_error=False, fill_value=None)
    print("Jc function interpolated")
    

# def return_interpolated_I_and_J():
#     """
#     returns globally stored Ic_interp and Jc_interp objects.
#     """
#     return Ic_interp, Jc_interp
