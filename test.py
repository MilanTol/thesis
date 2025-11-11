from config.config_cosmology import cosmo

import numpy as np
import matplotlib.pyplot as plt

z_vals = np.linspace(0, 3, 40)

rho_m_vals = cosmo.rho_m(z_vals)

plt.plot(z_vals, rho_m_vals)
plt.yscale("log")
plt.show()