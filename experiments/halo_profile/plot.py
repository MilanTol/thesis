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
figures_dir = root / "figures/halo_profile" 
data_dir = root / "data/halo_profile"


# load in our data
real = np.load(data_dir / "real.npz")
r = real['r']
dm_real = real['dm']
stellar_real = real['stellar']
gas_real = real['gas']

fourier = np.load(data_dir / "fourier.npz")
k = fourier['k']
dm_fourier = fourier['dm']
stellar_fourier = fourier['stellar']
gas_fourier = fourier['gas']

# plot real
plt.figure(figsize=(8,6))

plt.plot(r, 
        dm_real, 
        color='black',
        label="dm"
        )
plt.plot(r,
         stellar_real,
         color='red',
         label='stellar',
         linestyle='--')

plt.plot(r,
         gas_real,
         color='blue',
         label='gas',
         linestyle=':')

plt.xlabel('r [Mpc/h]')
plt.ylabel(r'weighted normalized density')
plt.legend()

plt.xscale("log")
plt.yscale("log")

plt.ylim(1e-3)

plt.savefig(figures_dir / "real.pdf",
            bbox_inches="tight")
plt.close()


# plot fourier
plt.figure(figsize=(8,6))

plt.plot(k, 
        dm_fourier, 
        color='black',
        label="dm"
        )
plt.plot(k,
         stellar_fourier,
         color='red',
         label='stellar',
         linestyle='--')

plt.plot(k,
         gas_fourier,
         color='blue',
         label='gas',
         linestyle=':')

plt.xlabel('k [h/Mpc]')
plt.ylabel(r'weighted normalized density')
plt.legend()

plt.xscale("log")
plt.yscale("log")

plt.ylim(1e-3)

plt.savefig(figures_dir / "fourier.pdf",
            bbox_inches="tight")
plt.close()
