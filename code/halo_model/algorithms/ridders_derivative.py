import numpy as np


def finite_difference(
    pair: np.ndarray, h: float
) -> float | np.ndarray:
    """
    A building block to compute derivative using finite differences

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : float | ndarray
        Derivative at x
    """
    res = 1/(2*h) * (pair[1] - pair[0])
    return res


def ridders_derivative(
    pairs:np.ndarray,
    h_init: float,
    d: float, 
    eps: float, 
) -> float | np.ndarray:
    """
    Computes derivative of given an array of pairs of estimates with decreasing distance to point of interest.

    Parameters
    ----------
    pairs : np.ndarray
        array of pairs with estimates of function: [ [f(x-h), f(x+h)], [f(x-h/2), f(x+h/2)], ... ]
    h_init : float
        Initial step size for finite difference
    d: float
        Factor by which to decrease h_init every iteration
    eps: float
        Target relative error

    Returns
    -------
    df : float | ndarray
        Derivative at x
    """
    # note that we only need to store the estimates that were currently adding (so i = iter)
    # and the previous estimates that we added (so i = iter - 1). 
    # So we only need 2 arrays of length max_iters.
    # this is more memory efficient than storing a triangular matrix

    # instantiate zero-arrays of length max_iters in which to store current estimate of D_iter_j's
    # and last added computation of D (so like D_iter-1_j)
    # so note that the indices correspond to j
    
    d_inv = 1/d # precompute d_inv so we only have to divide once when updating h
    max_iters = len(pairs)
    D = np.zeros(max_iters, dtype=object) # instantiate empty arrays
    D_last = np.zeros(max_iters, dtype=object)

    D[0] = finite_difference(pairs[0], h_init)
    D_last[0]  = D[0]

    #the best estimate always corresponds to D[j]
    best = D[0]
    err = np.inf #set error to infty to start with

    for i in range(1, max_iters):
        h_init *= d_inv # update h <-- h/d
        D[0] = finite_difference(pairs[i], h_init)
        for j in range(1, i+1): 
            D[j] = ( 
                (d**(2*j)*D[j-1] -  D_last[j-1])
                /
                (d**(2*j) - 1)
            ) #update D
        
        temp_err = np.abs(D[j] - D[j-1]) # compute of current iteration
        
        if temp_err < err: # if error grows, ignore
            err = temp_err
            best = D[j]
            if err < np.abs(eps*best): # check whether relative error is smaller than target 
                return best
            
            D_last[:i+1] = D[:i+1] # store current added estimates into D_last for next iteration

        else: #if error grows terminate early, cant cancel roundoff errors
            print("WARNING: target error-tolerance could not be achieved, " \
            f"error started growing at iteration {i}")
            return best
        
    return best