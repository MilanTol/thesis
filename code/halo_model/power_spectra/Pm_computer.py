

from halo_model.power_spectra.matter_power import MatterPower

from concurrent.futures import ProcessPoolExecutor
import numpy as np


def compute_for_k(args: tuple[MatterPower, float]) -> tuple[float, float, float, float, float, float, float]:
    Pm, k = args
    
    return (
        Pm.P_1h_ss(k),
        Pm.P_1h_sc(k),
        Pm.P_1h_self_c(k),
        Pm.P_1h_cc(k),
        Pm.P_2h_ss(k),
        Pm.P_2h_sc(k),
        Pm.P_2h_cc(k),
    )


def Pm_computer(Pm: MatterPower, k_vals, max_workers=16):
    """
    Returns a dictionary with MatterPower components. 
    This function computes the componenents at k_vals in parallel.

    Args
    :Pm: the matter power spectrum to be calculated
    :k_vals: the k_values at which to evaluate
    :max_workers: the max amount of workers used
    """
    
    dict_temp = {}

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        results = list(exe.map(compute_for_k, [(Pm, k) for k in k_vals]))

    results = np.array(results)

    dict_temp['P_1h_ss'] = results[:,0]
    dict_temp['P_1h_sc'] = results[:,1]
    dict_temp['P_1h_self_c'] = results[:,2]
    dict_temp['P_1h_cc'] = results[:,3]

    dict_temp['P_1h'] = (
        dict_temp['P_1h_ss'] +
        dict_temp['P_1h_sc'] +
        dict_temp['P_1h_self_c'] +
        dict_temp['P_1h_cc']
    )

    dict_temp['P_2h_ss'] = results[:,4]
    dict_temp['P_2h_sc'] = results[:,5]
    dict_temp['P_2h_cc'] = results[:,6]

    dict_temp['P_2h'] = (
        dict_temp['P_2h_ss'] +
        dict_temp['P_2h_sc'] +
        dict_temp['P_2h_cc']
    )

    dict_temp['P_tot'] = dict_temp['P_1h'] + dict_temp['P_2h']

    return dict_temp