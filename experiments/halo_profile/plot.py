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
data = np.load(data_dir / "contributions.npz")
r = data['r']
dm = data['dm']
stellar = data['stellar']
gas = data['gas']

# plot
plt.figure(figsize=(8,6))

plt.plot(r, 
            dm, 
            color='black',
            label="dm"
            )
plt.plot(r,
         stellar,
         color='red',
         label='stellar',
         linestyle='--')

plt.plot(r,
         gas,
         color='blue',
         label='gas',
         linestyle=':')

plt.xlabel('r [Mpc/h]')
plt.ylabel(r'weighted normalized density')
plt.legend()

plt.xscale("log")
plt.yscale("log")

plt.savefig(figures_dir / "smooth.pdf",
            bbox_inches="tight")
plt.close()
