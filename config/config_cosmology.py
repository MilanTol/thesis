#configure cosmology

# Cosmology parameters and functions
from colossus.cosmology import cosmology
from colossus.lss import mass_function

params = {
    'flat': True,
    'H0': 73.2,
    'Om0': 0.238,
    'Ob0': 0.042,
    'sigma8': 0.76,
    'ns': 0.958
}   

cosmology.addCosmology('myCosmo', **params)
cosmo = cosmology.setCosmology('myCosmo')