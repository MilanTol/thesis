"""Ready-made model configurations.

Assembling a :class:`~halo_model.power_spectra.matter_power.MatterPower` means
choosing a mass function, a bias, a clump mass function and three profiles, and
wiring them together consistently. 

Two standard setups are provided:

:func:`dark_matter_only`
    A halo is smooth dark matter plus dark-matter clumps.
:func:`with_baryons`
    A halo is smooth dark matter plus a central galaxy plus gas, with clumps
    carrying dark matter and satellite stars.
"""

from .config.config import Config
from .config.registry import COMPONENTS, MODELS

from .halos.mass_func.models.tinker2008 import MassFuncTinker2008_mod
from .halos.clump_mass_func.models.giocoli2010 import ClumpMassGiocoli2010_mod

from .halos.concentration.models.neto2007 import ConcentrationNeto2007
from .halos.concentration.models.scaled_clumps import ConcentrationScaledClumps
from .halos.concentration.models.scaled_distribution import ConcentrationScaledDistribution

from .halos.profile.models.gas_cored_powerlaw import ProfileGasCoredPowerLaw
from .halos.profile.models.nfw import ProfileNFW
from .halos.profile.models.stellar_truncated_powerlaw import (
    ProfileStellarTruncatedPowerLaw)

from .halos.profile.profile import CompositeProfile
from .halos.profile.weights import (
    CentralStarWeight, ClumpDarkMatterWeight, ClumpStarWeight, GasWeight,
    SmoothDarkMatterWeight, SmoothMassWeight, Unity)

from .halos.r_vir.models.SO import R_virSO

from .halos.shmr.models.moster2013 import (
    SHMRMoster2013, SHMRMoster2013Centrals)

from .power_spectra.matter_power import MatterPower


from dataclasses import dataclass

__all__ = ["Ingredients", "dark_matter_only", "with_baryons"]


@dataclass
class Ingredients:
    """The shared ingredients of a halo model, before profiles are composed.

    Attributes:
        cfg: the configuration these were built from.
        concentration: halo concentration-mass relation.
        concentration_clumps: concentration of clumps themselves.
        concentration_distribution: concentration of the clump spatial distribution.
        mass_func: halo mass function.
        clump_mass_func: clump mass function.
        r_vir: virial-radius relation.
    """

    cfg: Config
    concentration: ConcentrationNeto2007
    concentration_clumps: ConcentrationScaledClumps
    concentration_distribution: ConcentrationScaledDistribution
    mass_func: MassFuncTinker2008_mod
    clump_mass_func: ClumpMassGiocoli2010_mod
    r_vir: R_virSO

    @classmethod
    def build(cls, cfg: Config) -> "Ingredients":
        """Build the standard set of ingredients for a config."""
        concentration = ConcentrationNeto2007(cfg)
        return cls(
            cfg=cfg,
            concentration=concentration,
            concentration_clumps=ConcentrationScaledClumps(cfg, concentration),
            concentration_distribution=ConcentrationScaledDistribution(
                cfg, concentration),
            mass_func=MassFuncTinker2008_mod(cfg),
            clump_mass_func=ClumpMassGiocoli2010_mod(cfg),
            r_vir=R_virSO(cfg),
        )

    def nfw(self, concentration) -> ProfileNFW:
        """An NFW profile using the given concentration relation."""
        return ProfileNFW(concentration, self.r_vir)

    @property
    def smooth_dm_profile(self) -> ProfileNFW:
        """NFW profile for the smooth halo component."""
        return self.nfw(self.concentration)

    @property
    def clump_dm_profile(self) -> ProfileNFW:
        """NFW profile for an individual clump."""
        return self.nfw(self.concentration_clumps)

    @property
    def clump_distribution(self) -> ProfileNFW:
        """NFW profile for the spatial distribution of clumps within a halo."""
        return self.nfw(self.concentration_distribution)


def dark_matter_only(cfg: Config) -> MatterPower:
    """Build a dark-matter-only model.

    The smooth component holds the fraction of the halo mass that is not in
    clumps, and is evaluated at that reduced mass -- hence the same weight
    appears as both the amplitude weight and the mass-argument weight.

    Args:
        cfg: model configuration.
        
    Returns:
        A model ready to evaluate at ``cfg.z``.
    """
    parts = Ingredients.build(cfg)
    smooth_mass = SmoothMassWeight(parts.clump_mass_func)

    smooth_profile = CompositeProfile(
        parts.smooth_dm_profile,
        weights=[smooth_mass],
        arg_weights=[smooth_mass],
    )

    return MatterPower(
        cfg,
        mass_func=parts.mass_func,
        clump_mass_func=parts.clump_mass_func,
        smooth_profile=smooth_profile,
        clump_profile=parts.clump_dm_profile,
        clump_distribution=parts.clump_distribution
    )


def with_baryons(cfg: Config) -> MatterPower:
    """Build a model including baryons.

    The smooth component is dark matter plus a central galaxy plus gas; only
    the dark matter is evaluated at the reduced smooth mass, since the stellar
    and gas profiles are calibrated against the halo's total mass. Clumps carry
    dark matter and the satellite stars.

    Args:
        cfg: model configuration.

    Returns:
        A model ready to evaluate at ``cfg.z``.
    """
    parts = Ingredients.build(cfg)
    concentration = parts.concentration

    shmr = MODELS[COMPONENTS['shmr']][cfg.models['shmr']][1](cfg, concentration)
    shmr_centrals = MODELS[COMPONENTS['shmr_centrals']][cfg.models['shmr_centrals']][1](cfg, concentration)
    stellar_profile = MODELS[COMPONENTS['star_profile']][cfg.models['star_profile']][1](cfg, concentration)
    gas_profile = MODELS[COMPONENTS['gas_profile']][cfg.models['gas_profile']][1](cfg, shmr, concentration)

    smooth_profile = CompositeProfile(
        parts.smooth_dm_profile, stellar_profile, gas_profile,
        weights=[
            SmoothDarkMatterWeight(parts.clump_mass_func),
            CentralStarWeight(shmr_centrals),
            GasWeight(shmr),
        ],
        # Only the dark matter has its mass argument rescaled: it is evaluated
        # at the smooth dark-matter mass, so its concentration and virial
        # radius follow that mass rather than the halo total. The stellar and
        # gas profiles are calibrated against the halo's total mass and so are
        # left at M.
        arg_weights=[
            SmoothDarkMatterWeight(parts.clump_mass_func),
            Unity(),
            Unity(),
        ],
    )

    clump_profile = CompositeProfile(
        parts.clump_dm_profile, stellar_profile,
        weights=[
            ClumpDarkMatterWeight(),
            ClumpStarWeight(shmr, shmr_centrals),
        ],
    )

    model = MatterPower(
        cfg,
        mass_func=parts.mass_func,
        clump_mass_func=parts.clump_mass_func,
        smooth_profile=smooth_profile,
        clump_profile=clump_profile,
        clump_distribution=parts.clump_distribution,
    )
    # Kept accessible so callers can inspect the baryonic ingredients.
    model.shmr = shmr
    model.shmr_central = shmr_centrals
    model.stellar_profile = stellar_profile
    model.gas_profile = gas_profile
    return model
