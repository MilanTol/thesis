import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
root = Path(__file__).resolve().parents[2]

figures_dir = root / "figures/Pm_contributions"

# load in data
data_dir = root / "data/Pm_contributions/" 
ratios = np.load(data_dir / "ratios.npz")
k = ratios['k']

# plt.rcParams.update({
#     "font.size": 12,          # base font size
#     "axes.labelsize": 12,     # x and y labels
#     "axes.titlesize": 12,     # title
#     "legend.fontsize": 12,    # legend
#     "xtick.labelsize": 12,    # x tick labels
#     "ytick.labelsize": 12,    # y tick labels
# })

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


###
# create comparison plot with giocoli data
###

# interpolate giocoli data
from scipy.interpolate import interp1d

# -------------------------
# SS data
# -------------------------
x_ss, y_ss = np.loadtxt(
    data_dir / "giocoli_data/P_1h_ss.csv",
    delimiter=",",
    unpack=True
)

idx = np.argsort(x_ss)
x_ss = x_ss[idx]
y_ss = y_ss[idx]

f_ss = interp1d(
    np.log(x_ss), y_ss,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)

# -------------------------
# SC data
# -------------------------
x_sc, y_sc = np.loadtxt(
    data_dir / "giocoli_data/P_1h_sc.csv",
    delimiter=",",
    unpack=True
)

idx = np.argsort(x_sc)
x_sc = x_sc[idx]
y_sc = y_sc[idx]

f_sc = interp1d(
    np.log(x_sc), y_sc,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)

# -------------------------
# Self-correlation (c)
# -------------------------
x_self_c, y_self_c = np.loadtxt(
    data_dir / "giocoli_data/P_1h_self_c.csv",
    delimiter=",",
    unpack=True
)

idx = np.argsort(x_self_c)
x_self_c = x_self_c[idx]
y_self_c = y_self_c[idx]

f_self_c = interp1d(
    np.log(x_self_c), y_self_c,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)

# -------------------------
# CC data
# -------------------------
x_cc, y_cc = np.loadtxt(
    data_dir / "giocoli_data/P_1h_cc.csv",
    delimiter=",",
    unpack=True
)

idx = np.argsort(x_cc)
x_cc = x_cc[idx]
y_cc = y_cc[idx]

f_cc = interp1d(
    np.log(x_cc), y_cc,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)

# plot giocoli data and my data

ss_giocoli = f_ss(np.log(k))
sc_giocoli = f_sc(np.log(k))
self_c_giocoli = f_self_c(np.log(k))
cc_giocoli = f_cc(np.log(k))

# plot the residual of my computation with giocoli data

res_ss = (ratios['ss'] - ss_giocoli) / ss_giocoli
res_sc = (ratios['sc'] - sc_giocoli) / sc_giocoli
res_self_c = (ratios['self_c'] - self_c_giocoli) / self_c_giocoli
res_cc = (ratios['cc'] - cc_giocoli) / cc_giocoli


fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1,
    figsize=(4.5, 7),
    sharex=True,
        gridspec_kw={
        'height_ratios': [2, 2, 1.5],
    }
)
fig.subplots_adjust(
    hspace=0.02,
    left=0.15,
    right=0.98,
    top=0.97,
    bottom=0.10
)

# --- Giocoli data ---
ax1.plot(k, ss_giocoli, label=r'$P_{1h,ss}$', linestyle='-')
ax1.plot(k, sc_giocoli, label=r'$P_{1h,sc}$', linestyle='--')
ax1.plot(k, self_c_giocoli, label=r'$P_{1h,\mathrm{self-}c}$', linestyle='-.')
ax1.plot(k, cc_giocoli, label=r'$P_{1h,cc}$', linestyle=':')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.5, 5e3)
ax1.set_ylim(1e-3, None)
ax1.grid()
ax1.legend(fontsize='x-small')
ax1.set_ylabel(r'$P_{m,i}(k)/P_m(k)$')
ax1.set_title('Giocoli et al. (2010)')

# --- Your data ---
ax2.plot(k, ratios['ss'], label=r'$P_{1h,ss}$', linestyle='-')
ax2.plot(k, ratios['sc'], label=r'$P_{1h,sc}$', linestyle='--')
ax2.plot(k, ratios['self_c'], label=r'$P_{1h,\mathrm{self-}c}$', linestyle='-.')
ax2.plot(k, ratios['cc'], label=r'$P_{1h,cc}$', linestyle=':')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(1e-3, None)
ax2.set_xlim(0.5, 5e3)
ax2.grid()
ax2.legend(fontsize='x-small')
ax2.set_ylabel(r'$P_{m,i}(k)/P_m(k)$')
ax2.set_title('This work')

# --- Residuals ---
ax3.plot(k, res_ss,     label=r'$P_{1h,ss}$', linestyle='-')
ax3.plot(k, res_sc,     label=r'$P_{1h,sc}$', linestyle='--')
ax3.plot(k, res_self_c, label=r'$P_{1h,\mathrm{self-}c}$', linestyle='-.')
ax3.plot(k, res_cc,     label=r'$P_{1h,cc}$', linestyle=':')

ax3.axhline(0, color='k', lw=0.8, alpha=0.5)

ax3.set_xscale('log')
ax3.set_ylim(-0.3, 0.35)
ax3.set_xlim(0.5, 5e3)

ax3.grid()
ax3.legend(fontsize='x-small', ncol=2)

ax3.set_xlabel(r'$k \; [h/\mathrm{Mpc}]$')
ax3.set_ylabel(r'$(\Delta-\Delta_\mathrm{Giocoli})/\Delta_\mathrm{Giocoli}$')
ax3.set_title('Residuals')

plt.tight_layout()
plt.savefig("Pm_1h_ratios_comparison.pdf")
