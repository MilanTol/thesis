"""The angular (convergence) power spectrum, via the Limber approximation."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy import integrate

from ..algorithms.grids import LogInterpolator2D, tabulate_grid
from ..config.config import Config
from .matter_power import COMPONENTS, MatterPower
MATTER_COMPONENTS = COMPONENTS
__all__ = ["AngularPower", "SPEED_OF_LIGHT_KM_S"]
COMPONENTS = ("C_1h_ss", "C_1h_sc", "C_1h_self_c", "C_1h_cc")

#: Speed of light in km/s, matching the units Colossus reports H in.
SPEED_OF_LIGHT_KM_S = 299792.458


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

    Note:
        Building this object constructs one matter-power model per redshift,
        so it is the most expensive object in the package.
    """

    def __init__(self, cfg: Config,
                 build_model: Callable[[Config], MatterPower]):
        self.cfg = cfg
        self.build_model = build_model

        z_grid = cfg.z_grid()
        k_grid = cfg.k_grid()

        if cfg.z_sources > cfg.z_max:
            warnings.warn(
                f"z_sources={cfg.z_sources} lies beyond z_max={cfg.z_max}, so "
                "the line-of-sight integral extrapolates P(k, z) in redshift. "
                "Raise z_max to cover the source plane.",
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
        return sum(self.P(name, k, z) for name in COMPONENTS)

    def P_2h(self, k, z):
        """The two-halo term at ``(k, z)``."""
        return self.P("P_2h", k, z)

    def P_tot(self, k, z):
        """The total matter power spectrum at ``(k, z)``."""
        return self.P_1h(k, z) + self.P_2h(k, z)

    # ------------------------------------------------------------------
    # projection
    # ------------------------------------------------------------------

    def lensing_kernel(self, z):
        """Lensing efficiency for sources at a single plane ``cfg.z_sources``.

        Args:
            z: lens redshift.

        Returns:
            The kernel ``W(z)``, zero beyond the source plane.
        """
        cfg = self.cfg
        cosmo = cfg.cosmo   # also activates it for the Colossus globals

        prefactor = (1.5 * cosmo.Om0 * cosmo.H0 ** 2
                     / SPEED_OF_LIGHT_KM_S ** 2)
        if np.any(np.asarray(z) >= cfg.z_sources):
            # Past the source plane there is nothing left to lens. Without
            # this, Colossus is handed a negative redshift and returns a
            # negative distance, so the kernel silently goes negative.
            return np.zeros_like(np.asarray(z, dtype=float))

        # The middle factor is the comoving distance *between* lens and
        # source. Writing it as chi(z_s - z) -- the distance out to a redshift
        # difference -- is not the same thing, because chi is non-linear in z;
        # it overstates the ratio by 1.06x at z=0.1 rising to 2.75x at z=1.9,
        # and the kernel is squared in the projection.
        distance_ratio = (
            cosmo.comovingDistance(z_max=z)
            * cosmo.comovingDistance(z_min=z, z_max=cfg.z_sources)
            / cosmo.comovingDistance(z_max=cfg.z_sources))
        return prefactor * distance_ratio * (1 + z)

    def project(self, ell, P, weight=None, epsrel: float | None = None) -> float:
        """Project a 3-D spectrum to an angular one under the Limber approximation.

        Args:
            ell: angular multipole.
            P: the 3-D spectrum, called as ``P(k, z)``.
            weight: the kernel, called as ``w(z)``; defaults to
                :meth:`lensing_kernel`.
            epsrel: relative tolerance of the line-of-sight integral;
                defaults to ``cfg.epsrel``.

        Returns:
            ``C_ell`` for that spectrum.
        """
        cfg = self.cfg
        weight = self.lensing_kernel if weight is None else weight
        epsrel = cfg.epsrel if epsrel is None else epsrel

        def integrand(z):
            # Limber: C_l = int dz (c/H) W(z)^2 / chi^2 P(l/chi, z). The
            # lensing kernel already carries its full normalisation, so there
            # is no 2*pi here.
            distance = cfg.cosmo.comovingDistance(z_max=z)
            hubble = cfg.cosmo.Hz(z)          # km/s/Mpc, matching c above
            return (SPEED_OF_LIGHT_KM_S / hubble
                    * weight(z) ** 2 / distance ** 2 * P(ell / distance, z))

        value, _ = integrate.quad(integrand, 0, cfg.z_sources,
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


    def components(self, ell) -> dict[str, float]:
        """Every component of the spectrum at a single ``ell``.

        Returns:
            A dict with the four one-halo pieces, their sum ``C_1h``, the
            two-halo term ``C_2h`` and the total ``C_tot``.
        """
        values = {name: getattr(self, name)(ell) for name in COMPONENTS}
        values["P_1h"] = sum(values.values())
        values["P_2h"] = self.C_2h(ell)
        values["P_tot"] = values["P_1h"] + values["P_2h"]
        return values
 

def _spectrum_worker(args):
    """Evaluate every component of one model at one ``k``."""
    k, model = args
    return tuple(getattr(model, name)(k) for name in COMPONENTS) + (model.P_2h(k),)
