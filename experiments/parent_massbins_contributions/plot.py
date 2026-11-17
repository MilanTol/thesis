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

figures_dir = root / "figures/parent_massbins_contributions/"

# load in data
data_dir = root / "data/parent_massbins_contributions/" 
ratios = np.load(data_dir / "ratios.npz")

k = ratios['k']


MASSBINS = [
    # [8, 9],
    # [9, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
]


for massbin in MASSBINS:
    print(f'{massbin}')
    plt.plot(k, ratios[f'{massbin}'], label=f'{massbin}')
    
plt.yscale('log')
plt.xscale('log')

plt.ylim(1e-2, 2)
plt.xlim(1e-1, 1e2)

plt.xlabel(r'$k \; [h/\mathrm{Mpc}]$')
plt.ylabel(r'$P_{m, i}(k) / P_m(k)$')

plt.grid()
# plt.legend(fontsize='xx-small')
plt.legend()
plt.savefig(figures_dir / "Pm_1h_ratios.pdf", bbox_inches="tight")

