import numpy as np
import matplotlib
matplotlib.use("TkAgg")
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

figures_dir = root / "figures/Pm_WDM-CDM/"

# load in data
data_dir = root / "data/Pm_WDM-CDM/" 
ratios_dmo = np.load(data_dir / "ratios_dmo.npz")
ratios_bar = np.load(data_dir / "ratios_bar.npz")
k = ratios_dmo['k']


plt.plot(k, ratios_dmo['tot'], label = r'$P_{\text{DMO}}$', linestyle='-', color='black')
plt.plot(k, ratios_bar['tot'], label = r'$P_{\text{bar}}$', linestyle='--', color='red')

plt.yscale('log')
plt.xscale('log')

plt.ylim(0.9, 1.01)
plt.xlim(np.min(k), np.max(k))

plt.xlabel(r'$k \; [h/\mathrm{Mpc}]$')
plt.ylabel(r'$P_{WDM, i}(k) / P_{CDM, i}(k)$')

plt.grid()
# plt.legend(fontsize='xx-small')
plt.legend()
plt.savefig(figures_dir / "Pm_WDM-CDM_ratios.pdf", bbox_inches="tight")

