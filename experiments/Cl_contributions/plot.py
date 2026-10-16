import numpy as np
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

from pathlib import Path

# plt.rcParams.update({
#     "font.size": 12,          # base font size
#     "axes.labelsize": 12,     # x and y labels
#     "axes.titlesize": 12,     # title
#     "legend.fontsize": 12,    # legend
#     "xtick.labelsize": 12,     # x tick labels
#     "ytick.labelsize": 12,    # y tick labels
# })

root = Path(__file__).resolve().parents[2]

figures_dir = root / "figures/pm_contributions"

# load in data
data_dir = root / "data/pm_contributions/" 
ratios = np.load(data_dir / "ratios.npz")
k = ratios['k']


plt.plot(k, ratios['ss'], label = r'$P_{1h,ss}$', linestyle='-')
plt.plot(k, ratios['sc'], label = r'$P_{1h,sc}$', linestyle='--')
plt.plot(k, ratios['self_c'], label = r'$P_{1h,\mathrm{self-}c}$', linestyle='-.')
plt.plot(k, ratios['cc'], label = r'$P_{1h,cc}$', linestyle=':')

plt.yscale('log')
plt.xscale('log')

plt.ylim(1e-3, None)
plt.xlim(np.min(k), np.max(k))

plt.xlabel(r'$k \; [h/\mathrm{Mpc}]$')
plt.ylabel(r'$P_{m, i}(k) / P_m(k)$')

plt.grid()
# plt.legend(fontsize='xx-small')
plt.legend()
plt.savefig(figures_dir / "Pm_1h_ratios.pdf", bbox_inches="tight")