"""Mass-fraction weights for :class:`~halo_model.halos.base.profile.profile.CompositeProfile`.

Each weight answers "what fraction of this halo's mass does my component hold?"
and is called as ``weight(cosmo, M, z, **kwargs)``. They are small classes
rather than closures so that a model carrying them stays picklable, which is
what lets the (k, M) tabulation run across processes.
"""

from ..clump_mass_func.clump_mass_func import ClumpMassFunc
from ..shmr.shmr import SHMR
from colossus.cosmology import cosmology


__all__ = [
    "Unity",
    "SmoothMassWeight",
    "SmoothDarkMatterWeight",
    "CentralStarWeight",
    "GasWeight",
    "ClumpDarkMatterWeight",
    "ClumpStarWeight",
]


class Unity:
    """A weight of 1: the component holds the whole mass, or is not rescaled."""

    def __call__(self, cosmo: cosmology, M, z, **kwargs):
        return 1


class SmoothMassWeight:
    """Fraction of a halo's mass that is *not* bound up in clumps.

    The dark-matter-only model has no baryons to split off, so its smooth
    component carries this weight alone.
    """

    def __init__(self, clump_mass_func: ClumpMassFunc):
        self.clump_mass_func = clump_mass_func

    def __call__(self, cosmo: cosmology, M, z, **kwargs):
        return 1 - self.clump_mass_func.f(M)


class SmoothDarkMatterWeight:
    """Dark-matter fraction of the smooth (non-clumped) halo component.

    ``(1 - f_clump(M))`` of the halo is smooth, and ``(Om - Ob) / Om`` of that
    is dark matter.
    """

    def __init__(self, clump_mass_func: ClumpMassFunc):
        self.clump_mass_func = clump_mass_func

    def __call__(self, cosmo: cosmology, M, z, **kwargs):
        smooth_fraction = 1 - self.clump_mass_func.f(M)
        return smooth_fraction * (cosmo.Om(z) - cosmo.Ob(z)) / cosmo.Om(z)


class CentralStarWeight:
    """Stellar mass fraction locked up in the central galaxy."""

    def __init__(self, shmr_central: SHMR):
        self.shmr_central = shmr_central

    def __call__(self, cosmo: cosmology, M, z, **kwargs):
        return self.shmr_central(cosmo, M, z)


class GasWeight:
    """Gas fraction: all baryons that did not turn into stars."""

    def __init__(self, shmr: SHMR):
        self.shmr = shmr

    def __call__(self, cosmo: cosmology, M, z, **kwargs):
        return cosmo.Ob(z) / cosmo.Om(z) - self.shmr(cosmo, M, z)


class ClumpDarkMatterWeight:
    """Dark-matter fraction of a clump."""

    def __call__(self, cosmo: cosmology, M, z, **kwargs):
        return (cosmo.Om(z) - cosmo.Ob(z)) / cosmo.Om(z)


class ClumpStarWeight:
    """Stellar fraction of a clump: its satellites, i.e. total minus central."""

    def __init__(self, shmr: SHMR, shmr_central: SHMR):
        self.shmr = shmr
        self.shmr_central = shmr_central

    def __call__(self, cosmo: cosmology, m, z, **kwargs):
        return self.shmr(cosmo, m, z) - self.shmr_central(cosmo, m, z)
