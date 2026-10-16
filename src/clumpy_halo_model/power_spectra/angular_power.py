"""The angular (convergence) power spectrum, via the Limber approximation."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy import integrate
from scipy.integrate import cumulative_trapezoid, trapezoid

from ..algorithms.grids import LogInterpolator2D, tabulate_grid
from ..config.config import Config
from .matter_power import COMPONENTS as MATTER_COMPONENTS
from .matter_power import MatterPower

__all__ = ["AngularPower", "n_of_z_from_samples", "SPEED_OF_LIGHT_KM_S"]

#: Names of the projected one-halo pieces (the ``C_`` counterparts of
#: ``MATTER_COMPONENTS``). Kept separate from the 3-D names on purpose: the
#: two used to share the name ``COMPONENTS``, which made ``P_1h`` look up
#: ``C_*`` keys in the 3-D interpolator.
C_COMPONENTS = tuple("C_" + n.removeprefix("P_") for n in MATTER_COMPONENTS)

#: Speed of light in km/s, matching the units Colossus reports H in.
SPEED_OF_LIGHT_KM_S = 299792.458

#: Lowest redshift of the kernel grid (chi(0) = 0 would divide by zero).
_Z_MIN = 1e-4

#: Number of redshift points on which the lensing kernel is tabulated.
_KERNEL_GRID = 2000


def n_of_z_from_samples(z_samples, bins: int = 60, z_max: float | None = None):
    """Histogram redshift samples (e.g. from Flagship) into an ``n(z)``.

    Args:
        z_samples: one redshift per source galaxy.
        bins: number of histogram bins.
        z_max: upper edge of the histogram; defaults to the largest sample.

    Returns:
        ``(z, n)``: bin centres and the density, normalised to unit area.
    """
    z_samples = np.asarray(z_samples, dtype=float)
    z_max = float(z_samples.max()) if z_max is None else z_max
    counts, edges = np.histogram(z_samples, bins=bins, range=(0.0, z_max),
                                 density=True)
    return 0.5 * (edges[1:] + edges[:-1]), counts


class AngularPower:
    """Convergence power spectrum ``C_ell`` built from a 3-D matter spectrum.

    Each component of ``P(k)`` is evaluated at ``cfg.N_z`` redshifts, then
    interpolated over ``(k, z)`` and projected along the line of sight with the
    lensing kernel. Because every component is projected separately, the
    contribution of each one to ``C_ell`` can be inspected.

    Args:
        cfg: model configuration.
        build_model: a callable turning a config into a
            :class:`~halo_model.power_spectra.matter_power.MatterPower`,
            e.g. :func:`halo_model.models.with_baryons`.
        n_of_z: source redshift distribution as ``(z, n)`` arrays, e.g. from
            :func:`n_of_z_from_samples`. It need not be normalised. If
            ``None``, all sources sit on the single plane ``cfg.z_sources``.

    Note:
        Building this object constructs one matter-power model per redshift,
        so it is the most expensive object in the package.

        The distribution kernel assumes a flat universe, where the distance
        between lens and source is ``chi(z_s) - chi(z)``.
    """

    def __init__(self, cfg: Config,
                 build_model: Callable[[Config], MatterPower],
                 n_of_z: tuple | None = None):
        self.cfg = cfg
        self.build_model = build_model

        z_grid = cfg.z_grid()
        k_grid = cfg.k_grid()

        self._build_kernel(n_of_z)

        if self.z_top > cfg.z_max:
            warnings.warn(
                f"the sources reach z={self.z_top:.3g}, beyond "
                f"z_max={cfg.z_max}, so the line-of-sight integral "
                "extrapolates P(k, z) in redshift. Raise z_max to cover "
                "the sources.",
                RuntimeWarning, stacklevel=2)

        # One model per redshift. cfg.replace() rather than mutating cfg.z:
        # the previous implementation assigned to cfg.z in this loop, which
        # left the caller's config permanently pointing at the last redshift.
        self.models = [build_model(cfg.replace(z=z)) for z in z_grid]

        # The models are the second grid axis, so each task carries only the
        # one model it needs. Passing `self` here instead would drag the whole
        # list -- and `build_model` -- into every task, and would make a
        # non-picklable `build_model` (a lambda, say) fail to parallelise.
        tables = tabulate_grid(
            _spectrum_worker, (k_grid, self.models),
            n_outputs=len(MATTER_COMPONENTS) + 1, processes=cfg.processes)

        names = list(MATTER_COMPONENTS) + ["P_2h"]
        self._interpolators = {
            name: LogInterpolator2D(k_grid, z_grid, table, log_y=False)
            for name, table in zip(names, tables)
        }

    # ------------------------------------------------------------------
    # lensing kernel
    # ------------------------------------------------------------------

    def _build_kernel(self, n_of_z) -> None:
        """Tabulate the kernel and the background quantities on a z grid.

        Everything is in units of h: distances in Mpc/h, k in h/Mpc, matching
        Colossus' ``P(k)`` in (Mpc/h)^3. Mixing in Mpc and Mpc/h would leave
        ``C_ell`` off by a power of h.
        """
        cfg = self.cfg
        cosmo = cfg.cosmo   # also activates it for the Colossus globals
        h = cosmo.H0 / 100.0
        c = SPEED_OF_LIGHT_KM_S

        # 3/2 Om0 (H0/c)^2 with H0 = 100 h km/s/Mpc, in (h/Mpc)^2.
        prefactor = 1.5 * cosmo.Om0 * (100.0 / c) ** 2

        if n_of_z is None:
            z_top = float(cfg.z_sources)
        else:
            z_in, n_in = (np.asarray(a, dtype=float) for a in n_of_z)
            populated = z_in[n_in > 0]
            if populated.size == 0:
                raise ValueError("n_of_z has no positive entries")
            z_top = float(populated.max())

        zg = np.linspace(_Z_MIN, z_top, _KERNEL_GRID)
        chi = np.array([cosmo.comovingDistance(z_max=z) for z in zg])

        if n_of_z is None:
            # Single source plane. The middle factor is the comoving distance
            # *between* lens and source, not chi(z_s - z): chi is non-linear
            # in z, so the latter overstates the ratio (1.06x at z=0.1, 2.75x
            # at z=1.9) and the kernel is squared in the projection.
            chi_s = cosmo.comovingDistance(z_max=z_top)
            chi_ls = np.array([cosmo.comovingDistance(z_min=z, z_max=z_top)
                               for z in zg])
            efficiency = chi * chi_ls / chi_s
        else:
            n = np.interp(zg, z_in, np.clip(n_in, 0.0, None),
                          left=0.0, right=0.0)
            n /= trapezoid(n, zg)

            def tail(f):
                """int_z^z_top f dz' for every grid point."""
                return -cumulative_trapezoid(f[::-1], zg[::-1],
                                             initial=0.0)[::-1]

            # int_z n(z') (chi' - chi) / chi' dz'  =  tail(n) - chi tail(n/chi')
            efficiency = chi * (tail(n) - chi * tail(n / chi))

        W = prefactor * (1.0 + zg) * efficiency
        W[-1] = 0.0 if n_of_z is None else W[-1]

        hubble = np.array([cosmo.Hz(z) for z in zg])   # km/s/Mpc

        self.z_top = z_top
        self._z_tab = zg
        self._W_tab = W
        self._chi_tab = chi
        # dchi/dz = c / H in Mpc, so c h / H in Mpc/h.
        self._dchidz_tab = c * h / hubble

    def lensing_kernel(self, z):
        """Lensing efficiency ``W(z)`` for the configured source distribution.

        Args:
            z: lens redshift.

        Returns:
            The kernel, zero beyond the highest source redshift.
        """
        return np.interp(z, self._z_tab, self._W_tab, right=0.0)

    # ------------------------------------------------------------------
    # interpolated 3-D spectra
    # ------------------------------------------------------------------

    def P(self, name: str, k, z):
        """One named component of ``P(k, z)``.

        Args:
            name: one of the entries of
                :data:`~halo_model.power_spectra.matter_power.COMPONENTS`,
                or ``"P_2h"``.
            k: wavenumber [h/Mpc].
            z: redshift.

        Raises:
            KeyError: if ``name`` is not a tabulated component.
        """
        if name not in self._interpolators:
            raise KeyError(
                f"unknown component {name!r}; available: "
                f"{', '.join(sorted(self._interpolators))}")
        return self._interpolators[name](k, z)

    def P_1h_ss(self, k, z):
        """Smooth-smooth one-halo term at ``(k, z)``."""
        return self.P("P_1h_ss", k, z)

    def P_1h_sc(self, k, z):
        """Smooth-clump one-halo term at ``(k, z)``."""
        return self.P("P_1h_sc", k, z)

    def P_1h_self_c(self, k, z):
        """Clump self-correlation one-halo term at ``(k, z)``."""
        return self.P("P_1h_self_c", k, z)

    def P_1h_cc(self, k, z):
        """Clump-clump one-halo term at ``(k, z)``."""
        return self.P("P_1h_cc", k, z)

    def P_1h(self, k, z):
        """The full one-halo term at ``(k, z)``."""
        return sum(self.P(name, k, z) for name in MATTER_COMPONENTS)

    def P_2h(self, k, z):
        """The two-halo term at ``(k, z)``."""
        return self.P("P_2h", k, z)

    def P_tot(self, k, z):
        """The total matter power spectrum at ``(k, z)``."""
        return self.P_1h(k, z) + self.P_2h(k, z)

    # ------------------------------------------------------------------
    # projection
    # ------------------------------------------------------------------

    def project(self, ell, P, weight=None, epsrel: float | None = None):
        """Project a 3-D spectrum to an angular one under the Limber approximation.

        Args:
            ell: angular multipole, scalar or array.
            P: the 3-D spectrum, called as ``P(k, z)``.
            weight: the kernel, called as ``w(z)``; defaults to
                :meth:`lensing_kernel`.
            epsrel: relative tolerance of the line-of-sight integral;
                defaults to ``cfg.epsrel``.

        Returns:
            ``C_ell`` for that spectrum: a float for scalar ``ell``, an array
            otherwise.
        """
        ell = np.asarray(ell, dtype=float)
        if ell.ndim > 0:
            return np.array([self.project(float(l), P, weight, epsrel)
                             for l in ell.ravel()]).reshape(ell.shape)

        cfg = self.cfg
        weight = self.lensing_kernel if weight is None else weight
        epsrel = cfg.epsrel if epsrel is None else epsrel
        ell = float(ell)

        def integrand(z):
            # Limber: C_l = int dz (dchi/dz) W(z)^2 / chi^2 P((l+1/2)/chi, z).
            # The lensing kernel already carries its full normalisation, so
            # there is no 2*pi here.
            chi = np.interp(z, self._z_tab, self._chi_tab)
            dchidz = np.interp(z, self._z_tab, self._dchidz_tab)
            spectrum = float(np.squeeze(P((ell + 0.5) / chi, z)))
            return dchidz * weight(z) ** 2 / chi ** 2 * spectrum

        value, _ = integrate.quad(integrand, _Z_MIN, self.z_top,
                                  epsrel=epsrel, limit=200)
        return value

    def C(self, name: str, ell):
        """``C_ell`` for one named component of the matter spectrum."""
        return self.project(ell, lambda k, z: self.P(name, k, z))

    def C_1h_ss(self, ell):
        """Smooth-smooth contribution to ``C_ell``."""
        return self.C("P_1h_ss", ell)

    def C_1h_sc(self, ell):
        """Smooth-clump contribution to ``C_ell``."""
        return self.C("P_1h_sc", ell)

    def C_1h_self_c(self, ell):
        """Clump self-correlation contribution to ``C_ell``."""
        return self.C("P_1h_self_c", ell)

    def C_1h_cc(self, ell):
        """Clump-clump contribution to ``C_ell``."""
        return self.C("P_1h_cc", ell)

    def C_1h(self, ell):
        """One-halo contribution to ``C_ell``."""
        return self.project(ell, self.P_1h)

    def C_2h(self, ell):
        """Two-halo contribution to ``C_ell``."""
        return self.project(ell, self.P_2h)

    def C_tot(self, ell):
        """The total convergence power spectrum."""
        return self.C_1h(ell) + self.C_2h(ell)

    def components(self, ell) -> dict:
        """Every component of the spectrum at ``ell``.

        Returns:
            A dict with the four one-halo pieces (``C_1h_ss`` ...), their sum
            ``C_1h``, the two-halo term ``C_2h`` and the total ``C_tot``.
            Values are floats for scalar ``ell``, arrays otherwise.
        """
        values = {c_name: self.C(p_name, ell)
                  for c_name, p_name in zip(C_COMPONENTS, MATTER_COMPONENTS)}
        values["C_1h"] = sum(values.values())
        values["C_2h"] = self.C_2h(ell)
        values["C_tot"] = values["C_1h"] + values["C_2h"]
        return values


def _spectrum_worker(args):
    """Evaluate every component of one model at one ``k``."""
    k, model = args
    return (tuple(getattr(model, name)(k) for name in MATTER_COMPONENTS)
            + (model.P_2h(k),))