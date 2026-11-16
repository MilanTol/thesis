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

figures_dir = root / "figures/gas_comparison/"

# load in data
data_dir = root / "data/gas_comparison/" 
ratios = np.load(data_dir / "ratios.npz")
k = ratios['k']

plt.plot(k, ratios['dmo'], label=r'$dmo', linestyle='-', color='black')
plt.plot(k, ratios['debackere'], label = r'$debackere$', linestyle='-.', color='blue')
plt.plot(k, ratios['siegel'], label = r'$siegel$', linestyle='--', color='red')


plt.yscale('log')
plt.xscale('log')

plt.ylim(0.7, 1.2)
plt.xlim(np.min(k), np.max(k))

plt.xlabel(r'$k \; [h/\mathrm{Mpc}]$')
plt.ylabel(r'$P_{wdm}(k) / P_{cdm}(k)$')

plt.grid()
# plt.legend(fontsize='xx-small')
plt.legend()
plt.savefig(figures_dir / "Pm_ratios.pdf", bbox_inches="tight")
plt.close()


plt.plot(k, (ratios['debackere'] - ratios['siegel']) / ratios['debackere'], 
         label=r'(deb - sie) / deb', linestyle='-', color='black')


# plt.yscale('log')
plt.xscale('log')

# plt.ylim(-0.1, 0.1)
plt.xlim(np.min(k), np.max(k))

plt.xlabel(r'$k \; [h/\mathrm{Mpc}]$')
plt.ylabel(r'$residual$')

plt.grid()
# plt.legend(fontsize='xx-small')
plt.legend()
plt.savefig(figures_dir / "Pm_ratios_debsie_residual.pdf", bbox_inches="tight")