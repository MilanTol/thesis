from ..profile import Profile
from ....config.config import Config

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

# np.trapz was removed in numpy 2.0 (renamed np.trapezoid); support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Digitised Fig. 9c of Green et al. (2021), exported from WebPlotDigitizer:
# <repo root>/data/green2021/green2021_fig9.csv
_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_FILE = _ROOT / "data/green2021/green2021_fig9.csv"

# Dimensionless wavenumber grid on which u(k) is tabulated (k in units of 1/R_host).
# The upper end is limited by the radial resolution of the integral in _u_of_k:
# the integrand must stay resolved, i.e. k * dln(r) << 1 at r = r_out (k <~ 6e2).
_K_GRID = np.geomspace(1e-2, 1e2, 400)

# Inner radius (in units of R_host) of all radial integrals. Inside the first
# digitised point the profile is extrapolated as a power law, and the mass
# inside r < _R_MIN is negligible as long as the inner slope is shallower than -3.
_R_MIN = 1e-4


# ======================================================================
# Helper functions (module level, so the expensive table can be cached
# and shared between instances)
# ======================================================================

def _load_profiles(path, log_axes=True):
    """Read the WebPlotDigitizer export of Fig. 9c.

    File layout: row 1 holds the dataset names (here log10(m/M), one name per
    X column), row 2 holds "X,Y" labels, the rest is data. Datasets have
    different lengths, so shorter ones are padded with NaN.

    Args:
        path: path to the csv file.
        log_axes: True if the digitised x, y values are log10(r) and
            log10(rho); False if they are linear.

    Returns:
        dict {log10(m/M): (log10 r, log10 rho)}, with x sorted ascending
        and without duplicate x values.
    """
    with open(path) as f:
        names = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=2)

    profiles = {}
    for i in range(0, data.shape[1], 2):          # columns come in (X, Y) pairs
        x, y = data[:, i], data[:, i + 1]
        valid = ~np.isnan(x) & ~np.isnan(y)       # drop the NaN padding
        x, y = x[valid], y[valid]
        if not log_axes:
            x, y = np.log10(x), np.log10(y)
        x, idx = np.unique(x, return_index=True)  # sort and drop duplicates
        profiles[float(names[i])] = (x, y[idx])
    return profiles


def _make_rho(x, y, r_out, taper, n_fit=5):
    """Build a callable rho(r) from one digitised curve.

    Inside the data range the curve is linearly interpolated in log-log space.
    Inward of the first point it continues as a power law, with the slope
    fitted to the innermost ``n_fit`` points. Outward of ``r_out`` it is zero.

    Args:
        x, y: log10(r) and log10(rho) of the digitised curve.
        r_out: outer edge of the profile, in units of R_host.
        taper: width in dex of a cosine taper that smoothly brings the profile
            to zero just before ``r_out``. 0 gives a hard edge.
        n_fit: number of innermost points used for the inner power-law slope.

    Returns:
        function rho(r) with r in units of R_host, in arbitrary (unnormalised) units.
    """
    f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    s_in = np.polyfit(x[:n_fit], y[:n_fit], 1)[0]   # inner log-log slope
    x_edge = np.log10(r_out)

    def rho(r):
        r = np.asarray(r, float)
        lr = np.log10(r)
        # power law inward of the data, interpolation inside it
        ly = np.where(lr < x[0], y[0] + s_in * (lr - x[0]), f(lr))
        val = 10 ** ly
        if taper > 0:
            # weight 1 until (x_edge - taper), then falls to 0 at x_edge
            w = np.where(lr < x_edge - taper, 1.0,
                         0.5 * (1 + np.cos(np.pi * (lr - (x_edge - taper)) / taper)))
        else:
            w = 1.0
        return np.where(r <= r_out, val * w, 0.0)

    return rho


def _u_of_k(rho, k, r_out, n_r=20000):
    """Normalised Fourier transform of a spherical profile, u(0) = 1:

        u(k) = int rho(r) sin(kr)/(kr) r^2 dr  /  int rho(r) r^2 dr

    Both integrals run from _R_MIN to r_out and are done in ln r, which
    resolves the steep, power-law-like profile much better than a linear grid.

    Args:
        rho: callable rho(r), r in units of R_host.
        k: array of dimensionless wavenumbers (units of 1 / R_host).
        r_out: outer radius of the profile.
        n_r: number of ln r integration points.
    """
    ln_r = np.linspace(np.log(_R_MIN), np.log(r_out), n_r)
    r = np.exp(ln_r)
    w = rho(r) * r**3                          # rho r^2 dr = rho r^3 dln(r)
    j0 = np.sinc(np.outer(k, r) / np.pi)       # sin(kr)/(kr), shape (n_k, n_r)
    return _trapz(w * j0, ln_r, axis=1) / _trapz(w, ln_r)


def _rho_at(rho_funcs, log_m, lm, x):
    """Unnormalised profile rho(x) at mass ratio ``lm`` = log10(m/M).

    Between the digitised curves this interpolates linearly in log rho at fixed
    x (= geometric interpolation of rho). *Outside* the digitised range the same
    straight line in log rho vs log10(m/M) is simply continued, using the two
    outermost curves, i.e. a linear extrapolation in log rho.

    Args:
        rho_funcs: callables rho_j(x), one per digitised curve.
        log_m: ascending log10(m/M) of the digitised curves.
        lm: requested log10(m/M) (scalar); may lie outside ``log_m``.
        x: radius in units of R_host.
    """
    # index of the curve just below lm; clipped so that for lm outside the data
    # the two outermost curves are used (and the weight w leaves [0, 1])
    j = int(np.clip(np.searchsorted(log_m, lm, side="right") - 1, 0, len(log_m) - 2))
    w = (lm - log_m[j]) / (log_m[j + 1] - log_m[j])   # w<0 or w>1 -> extrapolation
    lo, hi = rho_funcs[j](x), rho_funcs[j + 1](x)
    both = (lo > 0) & (hi > 0)                        # both vanish beyond r_out
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(both, lo**(1 - w) * hi**w, 0.0)


@dataclass(frozen=True, eq=False)
class _GreenTable:
    """Everything derived from the csv; built once and cached (see _build_table)."""
    log_m: np.ndarray        # log10(m/M) of the *digitised* curves, ascending
    rho: tuple               # rho_j(x) callables of the digitised curves
    interp: RegularGridInterpolator   # u(log10(m/M), ln(k R_host)) on the extended grid
    lm_min: float            # range of log10(m/M) covered by the table (extended)
    lm_max: float
    k_min: float             # edges of the tabulated dimensionless k range
    k_max: float
    r_out: float


@functools.lru_cache(maxsize=None)
def _build_table(path, log_axes, r_out, taper, lm_min, lm_max):
    """Load the digitised curves and tabulate u(k) on an extended m/M grid.

    The grid has ~1 dex spacing and spans [lm_min, lm_max], which is wider than
    the digitised range [-10, -4]. The profile at each node comes from
    ``_rho_at`` (interpolated inside, extrapolated outside the data) and u(k)
    is computed from that profile by integration, so the Fourier-space and
    real-space profiles stay consistent in the extrapolated region.

    Cached on the arguments, so creating several profile objects (e.g. in a
    parameter sweep) does not repeat the integrals.
    """
    profiles = _load_profiles(path, log_axes)
    log_m = np.array(sorted(profiles))                       # e.g. [-10, -8, -6, -4]
    rho = tuple(_make_rho(*profiles[lm], r_out, taper) for lm in log_m)

    # the tabulated range must at least contain the digitised one
    lm_lo, lm_hi = min(lm_min, log_m[0]), max(lm_max, log_m[-1])
    n_nodes = max(int(np.ceil(lm_hi - lm_lo)) + 1, 2)        # ~1 dex spacing
    lm_nodes = np.linspace(lm_lo, lm_hi, n_nodes)

    u_all = np.array([
        _u_of_k(lambda r, lm=lm: _rho_at(rho, log_m, lm, r), _K_GRID, r_out)
        for lm in lm_nodes
    ])                                                       # (n_nodes, n_k)

    interp = RegularGridInterpolator(
        (lm_nodes, np.log(_K_GRID)), u_all,
        method="linear", bounds_error=False, fill_value=None,
    )
    return _GreenTable(log_m, rho, interp, lm_lo, lm_hi,
                       _K_GRID[0], _K_GRID[-1], r_out)


# ======================================================================
# The profile class
# ======================================================================

class ProfileGreen2021(Profile):
    """
    Spatial distribution of clumps (subhaloes) inside their host halo, taken
    from the digitised Fig. 9c of Green et al. (2021).

    The figure gives the radial profile for four clump-to-host mass ratios,
    log10(m/M) = -10, -8, -6, -4, with radius in units of the host radius.
    Between these curves we interpolate linearly in log rho (at fixed r); outside
    the digitised range the same trend is linearly *extrapolated* in log rho,
    out to ``lm_range`` (default log10(m/M) in [-16, 0]); beyond that the
    nearest edge of ``lm_range`` is used. Extrapolated shapes are not data:
    inspect them (see the notes on ``lm_range``).

    Normalisation: the profile integrates to 1 over the host volume,
        int 4 pi r^2 rho(r) dr = 1,   u(k -> 0) = 1,
    i.e. it is a unit-normalised distribution. The mass weights (how much mass
    sits in clumps) are applied elsewhere in the composite profile.

    Dependence on the clump mass: the profile depends on m/M, but the standard
    Profile interface only passes the host mass M. The clump mass can be passed
    as the keyword ``m`` (same units as M) to ``fourier`` / ``real``; if it is
    omitted, the fixed ``log_m_over_M`` given at construction is used.

    Units: radii and wavenumbers follow the rest of the pipeline (Mpc/h, h/Mpc).
    R_host is the radius of the halo mass definition ``cfg.massdef`` (e.g. R200c).
    This assumes Green et al. normalise their radii the same way; if they use
    another radius (e.g. R_vir), change ``_host_radius``.
    """

    def __init__(self, cfg: Config, *,
                 data_file=None, log_axes: bool = True,
                 r_out: float = 1.0, taper: float = 0.0,
                 lm_range=(-16.0, 0.0), **_context):
        """
        Args:
            cfg: model configuration (only ``massdef`` and ``cosmo`` are used).
            log_m_over_M: log10(m/M) used when no clump mass ``m`` is passed
                to ``fourier`` / ``real``. Default -6.0 is the middle of the
                tabulated range [-10, -4].
            data_file: csv with the digitised profiles; defaults to
                data/green2021/green2021_fig9.csv in the repository.
            log_axes: True if the csv holds log10(r), log10(rho).
            r_out: outer edge of the profile in units of R_host. The
                digitised profiles end at the host radius, so 1.0.
            taper: width (dex) of a cosine taper before ``r_out``. 0 gives a
                hard edge, which produces oscillations in u(k) at high k.
            lm_range: (lowest, highest) log10(m/M) up to which the profile is
                extrapolated beyond the digitised range [-10, -4]. The default
                covers m_min = 1e3 in hosts up to 1e16 (about -13) and clumps
                up to the host mass (0). Outside this range log10(m/M) is
                clipped. A linear extrapolation in log rho keeps drifting
                (slopes flatten or steepen without limit), so keep the range
                no wider than your integration needs.
            **_context: other objects the pipeline may pass (r_vir,
                concentration, ...); not needed here, so ignored.
        """
        self.cfg = cfg

        # --- host radius definition, parsed from cfg.massdef ("200c", "200m", ...)
        massdef = cfg.massdef
        if len(massdef) < 2 or massdef[-1] not in ("c", "m"):
            raise ValueError(f"unsupported mass definition {massdef!r}; "
                             "expected e.g. '200c' or '200m'")
        self._delta = float(massdef[:-1])     # overdensity, e.g. 200
        self._ref = massdef[-1]               # 'c' critical / 'm' mean matter density

        # --- tabulated u(k | m/M); built once per (file, settings) and cached
        path = str(data_file if data_file is not None else DEFAULT_DATA_FILE)
        self._table = _build_table(path, bool(log_axes), float(r_out), float(taper),
                                   float(lm_range[0]), float(lm_range[1]))

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    def _host_radius(self, M, z):
        """Host radius in Mpc/h: mean enclosed density = delta * rho_ref(z)."""
        cosmo = self.cfg.cosmo
        # same unit convention (*1e9) as the gas profiles in this pipeline
        rho_ref = (cosmo.rho_c(z=z) if self._ref == "c" else cosmo.rho_m(z=z)) * 1e9
        return (3 * np.asarray(M, float) / (4 * np.pi * self._delta * rho_ref))**(1 / 3)

    def _log_m_over_M(self, M, m=None):
        """log10(m/M), clipped to the (extended) range covered by the table."""
        if m is None:
            raise ValueError(f"Green2021 clump distribution requires m as kwarg")
        else:
            lm = np.log10(np.asarray(m, float) / np.asarray(M, float))
        # extrapolation is allowed (see lm_range), but not without limit
        return np.clip(lm, self._table.lm_min, self._table.lm_max)

    def _rho_dimless(self, x, lm):
        """Unnormalised rho(x) at mass ratio lm (scalar), x = r / R_host.

        Interpolated / extrapolated in log10(m/M) by ``_rho_at``.
        """
        return _rho_at(self._table.rho, self._table.log_m, lm, x)

    # ------------------------------------------------------------------
    # Profile interface
    # ------------------------------------------------------------------

    def _fourier(self, cosmo, k, M, z, **kwargs):
        """Unit-normalised Fourier profile u(k | M, m), u(0) = 1.

        Args:
            k: wavenumber(s) in h/Mpc.
            M: host halo mass in cfg.massdef [Msun/h].
            z: redshift.
            m: (keyword, optional) clump mass [Msun/h]; may be far outside
                the digitised m/M range (extrapolated). If omitted the
                constructor value ``log_m_over_M`` is used.
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))

        # The table is in dimensionless kR, so convert with the host radius.
        kR = k * self._host_radius(M, z)
        lm = self._log_m_over_M(M, kwargs.get("m"))
        kR, lm = np.broadcast_arrays(kR, lm)     # allows array k, M and/or m

        # Below the table (kR < k_min) u is already ~1, so clip. Above it
        # (kR > k_max) the profile has long been averaged out (|u| << 1), so
        # we set u = 0 there rather than extrapolate an oscillating tail.
        in_range = kR <= self._table.k_max
        kR_clipped = np.clip(kR, self._table.k_min, self._table.k_max)

        pts = np.stack([lm, np.log(kR_clipped)], axis=-1)
        u = np.where(in_range, self._table.interp(pts), 0.0)

        # same return convention as the other profiles: float for a single k
        return u if u.size > 1 else float(u.ravel()[0])

    def real(self, cosmo, r, M, z, **kwargs):
        """Unit-normalised density in real space [(Mpc/h)^-3]:
        int 4 pi r^2 real(r) dr = 1 over the host volume.

        Args:
            r: radius/radii in Mpc/h.
            M: host halo mass in cfg.massdef [Msun/h]; scalar.
            z: redshift.
            m: (keyword, optional) clump mass [Msun/h]; scalar.
        """
        r = np.atleast_1d(np.asarray(r, dtype=float))
        R = float(self._host_radius(M, z))
        lm = float(self._log_m_over_M(M, kwargs.get("m")))

        # shape of the profile at the requested radii
        rho = self._rho_dimless(r / R, lm)

        # volume integral of the same shape, used to normalise to 1
        x = np.geomspace(_R_MIN, self._table.r_out, 4000)
        norm = 4 * np.pi * _trapz(self._rho_dimless(x, lm) * x**3, np.log(x))

        # rho(r) d^3r = rho(x) R^3 d^3x  =>  divide by R^3 * norm
        return rho / (R**3 * norm)