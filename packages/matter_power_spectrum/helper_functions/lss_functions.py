#imports
from colossus.lss.mass_function import massFunction
from colossus.lss.bias import haloBias


def halo_mass_function(config, M):
    """compute halo mass function"""
    z = config['z']
    Delta_vir = config['Delta_vir']
    return massFunction(M, z=z, mdef = f'{Delta_vir}m', model = 'tinker08', q_out='dndlnM') / M #divide by M for jacobian dlnM/dM


def halo_bias(config, M):
    """compute halo bias"""
    z = config['z']
    Delta_vir = config['Delta_vir']    
    return haloBias(M, z=z, mdef = f'{Delta_vir}m', model='tinker10')