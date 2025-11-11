# Interpolates all necessary functions

from . import halo_functions as halo
from . import I_and_J_integral_functions_interpolated as I_and_J

def interpolate():
    """
    interpolate all necessary helper functions.
    """
    halo.interpolate_f_sub()
    I_and_J.interpolate_I_and_J()