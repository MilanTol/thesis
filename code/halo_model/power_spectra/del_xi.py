import copy
import numpy as  np

from halo_model.config.config import Config
from halo_model.algorithms.ridders_derivative import ridders_derivative
from halo_model.power_spectra.xi_computer import compute_xi


def del_xi(cfg:Config, param: str, h_init:float, n_iters:int):
    """
    Computes derivative (with Ridders) of the xi+ weak lensing observable 
    wrt a parameter (param) set in config.

    Args:
        cfg (Config): 
            config at which to take derivative
        param (str): 
            parameter over which to take derivative (eg. beta -> slope of clump mass function)
        h_init (float): 
            initial offset of parameter.
        n_iters (int): 
            number of finite differences that Ridders uses. 
            This function computes (2*n_iters) xi+ arrays.

    Returns:
        derivs (np.ndarray): 
            array containing derivatives at each theta_bin
    """

    # Compute power spectrum (at various redshifts):

    h = h_init

    # Shape: [n_iters, N_theta] for low and up separately
    xi_lows = []
    xi_ups  = []
            
    for i in range(n_iters):
        
        print(f"computing the {i}th pair for Ridders")
        
        cfg_low = copy.deepcopy(cfg)
        cfg_up = copy.deepcopy(cfg) 
        
        value = getattr(cfg, param)

        setattr(cfg_low, param, value - h)
        setattr(cfg_up, param, value + h)
                
        xi_up = compute_xi(cfg_up)
        xi_low = compute_xi(cfg_low)
                
        xi_lows.append(xi_low)   # shape: [N_theta]
        xi_ups.append(xi_up)
            
        h *= 0.5
        
    # pairs[i, theta_idx] = [xi_low, xi_up] at iteration i
    # ridders_derivative expects pairs[:, theta_idx] -> shape [n_iters, 2]
    pairs = np.array([
        [[xi_lows[i][t], xi_ups[i][t]] for t in range(cfg.N_theta)]
        for i in range(n_iters)
    ])  # shape: [n_iters, N_theta, 2]
        
    
    # Compute derivative for each theta angle
    derivs = np.array([
        ridders_derivative(pairs[:, t, :], h_init=h_init, d=2, eps=1e-9)
        for t in range(cfg.N_theta)
    ])  
    
    return derivs



