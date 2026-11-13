import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

from pathlib import Path
from scipy.interpolate import interp1d, RegularGridInterpolator

root = Path(__file__).resolve().parents[2]
data_dir = root / "data/green2021"

# np.trapz was removed in numpy 2.0
trapz = getattr(np, "trapezoid", None) or np.trapz

LOG_AXES = True      # CSV x,y are log10(r), log10(rho); False if linear
R_OUT = 1.0          # outer edge of the digitized profiles
TAPER = 0.0          # dex of cosine taper before R_OUT; 0 = hard edge


def load_profiles(path):
    """WPD export: row 1 = dataset names, row 2 = X,Y. Returns {log10(m/M): (log r, log rho)}."""
    with open(path) as f:
        names = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=2)

    profiles = {}
    for i in range(0, data.shape[1], 2):
        x, y = data[:, i], data[:, i + 1]
        m = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[m], y[m]
        if not LOG_AXES:
            x, y = np.log10(x), np.log10(y)
        x, idx = np.unique(x, return_index=True)   # sort + drop duplicates
        profiles[float(names[i])] = (x, y[idx])
    return profiles


def make_rho(x, y, n_fit=5):
    """rho(r): log-log linear inside the data, power law inward, zero beyond R_OUT."""
    f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    s_in = np.polyfit(x[:n_fit], y[:n_fit], 1)[0]   # inner slope
    xe = np.log10(R_OUT)

    def rho(r):
        r = np.asarray(r, float)
        lr = np.log10(r)
        ly = np.where(lr < x[0], y[0] + s_in * (lr - x[0]), f(lr))
        val = 10 ** ly
        if TAPER > 0:
            w = np.where(lr < xe - TAPER, 1.0,
                0.5 * (1 + np.cos(np.pi * (lr - (xe - TAPER)) / TAPER)))
        else:
            w = 1.0
        return np.where(r <= R_OUT, val * w, 0.0)
    return rho


def u_of_k(rho, k, rmin=1e-4, Nr=20000):
    """u(k) = int rho j0(kr) r^2 dr / int rho r^2 dr  (integrated in ln r)."""
    lnr = np.linspace(np.log(rmin), np.log(R_OUT), Nr)
    r = np.exp(lnr)
    w = rho(r) * r**3
    j0 = np.sinc(np.outer(k, r) / np.pi)            # sin(kr)/(kr)
    return trapz(w * j0, lnr, axis=1) / trapz(w, lnr)


# ---- build ----
profiles = load_profiles(data_dir / "green2021_fig9.csv")
log_m = np.array(sorted(profiles))                  # [-10, -8, -6, -4]
rhos = {lm: make_rho(*profiles[lm]) for lm in log_m}

k = np.geomspace(1e-2, 1e2, 400)
Fk_all = np.array([u_of_k(rhos[lm], k) for lm in log_m])   # (n_mass, n_k)

_interp = RegularGridInterpolator(
    (log_m, np.log(k)), Fk_all,
    method="linear", bounds_error=False, fill_value=None,
)


def u_sub(k_in, log10_m_over_M):
    """Normalized Fourier profile u(k | m/M). Broadcasts over both arguments."""
    k_in, lm = np.broadcast_arrays(np.asarray(k_in, float),
                                   np.asarray(log10_m_over_M, float))
    return _interp(np.stack([lm, np.log(k_in)], axis=-1))


# ---- checks ----
if __name__ == "__main__":
    r = np.geomspace(10**-1.5, R_OUT, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lm in log_m:
        x, y = profiles[lm]
        ax[0].loglog(r, rhos[lm](r), label=f"{lm:g}")
        ax[0].loglog(10**x, 10**y, "k.", ms=3)
        ax[1].semilogx(k, Fk_all[list(log_m).index(lm)], label=f"{lm:g}")
    ax[0].set(xlabel="r", ylabel=r"$\rho$"); ax[1].set(xlabel="k", ylabel="u(k)")
    ax[0].legend(); ax[1].legend()
    plt.tight_layout(); plt.show()