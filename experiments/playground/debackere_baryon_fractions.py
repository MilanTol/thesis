"""
Recreate the Debackere et al. baryon fractions (Figure 9) with ProfileGasDebackere:
f_bar_200m vs m_200m_obs, coloured by gamma (0..3).
"""

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from scipy.optimize import brentq

from clumpy_halo_model.config.config import Config
from clumpy_halo_model.halos.concentration.models.neto2007 import ConcentrationNeto2007
from clumpy_halo_model.halos.profile.models.gas_debackere2022 import ProfileGasDebackere2022
from clumpy_halo_model.halos.shmr.models.moster2013 import SHMRMoster2013

from pathlib import Path
root = Path(__file__).resolve().parents[2]

# load config file
cfg = Config.from_yaml(root/"experiments/playground/config.yaml")

cosmo, z = cfg.cosmo, cfg.z
concentration = ConcentrationNeto2007(cfg)  
shmr = SHMRMoster2013(cfg, concentration)    

prof = ProfileGasDebackere2022(cfg, shmr, concentration)
base_params = cfg.p("gas_profile")

rho_c = cosmo.rho_c(z=z) * 1e9               # same unit convention as the profile

masses_200m = np.geomspace(1e10, 1e15, 50)  # m500c [Msun/h]
gammas = np.linspace(0, 3, 9)
cmap, norm = cm.plasma_r, colors.Normalize(0, 3)   # yellow = 0, dark blue = 3


def halo_mass_from_M200(M200m_target):
    """Halo mass in cfg.massdef whose M200m equals the target (root find)."""
    f = lambda logM: np.log10(float(prof._convert(cosmo, 10**logM, z, "200m"))) \
        - np.log10(M200m_target)
    if cfg.massdef == "200m":
        return M200m_target
    return 10**brentq(f, np.log10(M200m_target), np.log10(M200m_target) + 1.5)


for gamma in gammas:
    fig = plt.figure(figsize=(10, 7))
    

# for ax, M500 in zip(axes, masses_500c):
#     M = halo_mass_from_M500(M500)
#     M200m = prof._convert(cosmo, M, z, "200m")
#     M200c = prof._convert(cosmo, M, z, "200c")
#     r500 = prof.R_delta(M500, 500, "c", z)
#     r200m = prof.R_delta(M200m, 200, "m", z)
#     r200c = prof.R_delta(M200c, 200, "c", z)
#     f_b = cosmo.Ob(z) / cosmo.Om(z)

#     x = np.geomspace(1e-2, 6, 600)
#     r = x * r200m

#     for g in gammas:
#         prof.p = dataclasses.replace(base_params, gamma=float(g))
#         rho = prof.real(cosmo, r, M, z) / rho_c
#         inside = r < r200m
#         color = cmap(norm(g))
#         ax.plot(x[inside], np.where(rho[inside] > 0, rho[inside], np.nan),
#                 color=color, lw=1.2)
#         ax.plot(x[~inside], np.where(rho[~inside] > 0, rho[~inside], np.nan),
#                 color=color, lw=1.2, ls="--")

#         # diagnostics: does the profile do what it should?
#         rr = np.geomspace(1e-4, 1e3, 6000)
#         rh = prof.real(cosmo, rr, M, z)
#         m_tot = np.trapz(4 * np.pi * rr**2 * rh, rr)
#         m_500 = np.trapz(4 * np.pi * rr**2 * rh * (rr < r500), rr)
#         r_h = rr[rh > 0].max()
#         print(f"{np.log10(M500):8.1f} {g:6.3f} {m_500 / M500:9.3f} "
#               f"{r_h / r200m:10.2f} {m_tot / (f_b * M):12.3f}")

#     ax.axvline(r500 / r200m, color="purple", ls="--", lw=0.9)
#     ax.axvline(r200c / r200m, color="navy", ls="--", lw=0.9)
#     ax.set(xscale="log", yscale="log", xlim=(1e-2, 6), ylim=(1e-2, 1e4),
#            title=rf"$m_{{500c}}=10^{{{np.log10(M500):.1f}}}\,h^{{-1}}M_\odot$")
#     ax.set_xlabel(r"$r/r_{200m}$")

# axes[0].set_ylabel(r"$\rho_{\rm gas}(r)/\rho_c$")
# fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label=r"$\gamma$")
# plt.show()