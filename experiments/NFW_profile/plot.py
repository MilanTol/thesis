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
figures_dir = root / "figures/NFW_profile" 
data_dir = root / "data/NFW_profile"


# load in our data
data_real = np.load(data_dir / "real.npz")
r = data_real['r']
real = data_real['real']

data_fourier = np.load(data_dir / "fourier.npz")
k = data_fourier['k']
fourier = data_fourier['fourier']
fourier_analytic = data_fourier['fourier_analytic']


# plot real

plt.plot(r, 
            real, 
            color='black',
            )

plt.xlabel('r [Mpc/h]')
plt.ylabel(r'normalized density')

plt.xscale("log")
plt.yscale("log")

plt.savefig(figures_dir / "real.pdf",
            bbox_inches="tight")
plt.close()

# plot fourier

plt.plot(k, 
            fourier, 
            color='black',
            label='numeric'
            )
plt.plot(k, 
            fourier_analytic, 
            color='red',
            label='analytic'
            )

plt.xlabel('k [h/Mpc]')
plt.ylabel(r'normalized density')

plt.xscale("log")
plt.yscale("log")

plt.legend()

plt.savefig(figures_dir / "fourier.pdf",
            bbox_inches="tight")
plt.close()
