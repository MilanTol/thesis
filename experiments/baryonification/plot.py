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
figures_dir = root / "figures/baryonification" 
data_dir = root / "data/baryonification" 


# load in our data
our_data = np.load(data_dir / "our_reaction.npz")
k = our_data['k']
reaction = our_data['reaction']


# load in flamingo data and interpolate:
from scipy.interpolate import interp1d

data = np.genfromtxt(
    data_dir / "flamingo_reaction.csv",
    delimiter=",",
    skip_header=2,
    filling_values=np.nan,
)

x_f8 = data[:, 0]
y_f8 = data[:, 1]

mask0 = ~np.isnan(data[:, 2]) & ~np.isnan(data[:, 3])
x_f0 = data[mask0, 2]
y_f0 = data[mask0, 3]

f_f8 = interp1d(
    np.log(x_f8), y_f8,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)

f_f0 = interp1d(
    np.log(x_f0), y_f0,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)

f8_flamingo_reaction = f_f8(np.log(k))
f0_flamingo_reaction = f_f0(np.log(k))


plt.plot(k, 
            reaction, 
            label=f'This work',
            color='black',
            linestyle='-',
            )

plt.plot(k, 
            f8_flamingo_reaction, 
            label=r'Flamingo - $8\sigma$',
            color='red',
            linestyle='--')

plt.plot(k,
            f0_flamingo_reaction, 
            label=r'Flamingo - $0\sigma$',
            color='blue',
            linestyle='-.')

plt.legend(loc="lower left")
plt.ylim(0.7, 1.1)
plt.xlim(1e-1, 1e2)
plt.xlabel('k [h/Mpc]')
plt.ylabel(r'$P(k) / P_\mathrm{DMO}(k)$')

# plt.title("Power spectrum")
plt.xscale("log")

plt.savefig(figures_dir / "baryonification_comparison.pdf",
            bbox_inches="tight")
