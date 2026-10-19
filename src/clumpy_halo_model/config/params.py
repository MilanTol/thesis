R"""
Parameter sections for every selectable model.

Each model gets one frozen dataclass holding its parameters, with no default
values: every value comes from config.yaml. COMPONENTS maps
``component -> {model name -> parameter dataclass}``. 

To add a model: 
write its dataclass, add one line to COMPONENTS, 
add its section (and selection) to config.yaml, 
and register its implementation class in the registry.

This module must stay free of heavy imports: Config imports it, and the
profile classes import Config.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ModelParams", "NoParams", "GasSiegel2025", "GasDebackere2022",
    "COMPONENTS", "build_section",
]


@dataclass(frozen=True)
class ModelParams:
    """Base class of every model's parameter section."""

    def validate(self) -> None:
        """Raise ValueError for invalid values.

        Only called for the *selected* model of each component, so unused
        sections can hold anything that type-checks.
        """


@dataclass(frozen=True)
class NoParams(ModelParams):
    """For models without free parameters."""


# ---------------------------------------------------------------------
# gas profile
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GasSiegel2025(ModelParams):
    """Cored double power law, Siegel et al. 2025 (Table 1)."""
    logM_c: float
    theta_ej: float
    mu: float
    gamma: float
    delta: float


@dataclass(frozen=True)
class GasDebackere2022(ModelParams):
    """Beta profile + power law + constant density out to r_h (Debackere et al.).

    f_gas,500c(m500c) follows eq. 25-26: a tanh turnover at mass ``10**logM_t``
    with sharpness ``alpha_t``.
    """
    gamma: float        # outer power-law slope, 0 <= gamma <= 3
    logM_t: float      # turnover mass of f_gas,500c [log10 Msun/h]
    alpha_t: float      # sharpness of the turnover
    rc_frac: float      # core radius in units of r500c
    beta_gas: float     # beta-profile slope

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 3.0:
            raise ValueError(f"GasDebackere2022.gamma must be in [0, 3], got {self.gamma}")
        if self.alpha_t <= 0:
            raise ValueError(f"GasDebackere2022.alpha_t must be > 0, got {self.alpha_t}")
        if self.rc_frac <= 0:
            raise ValueError(f"GasDebackere2022.rc_frac must be > 0, got {self.rc_frac}")


# ---------------------------------------------------------------------
# component table
# ---------------------------------------------------------------------

COMPONENTS: dict[str, dict[str, type[ModelParams]]] = {
    "gas_profile": {
        "siegel2025": GasSiegel2025,
        "debackere2022": GasDebackere2022,
    },
}

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

_CASTS = {"float": float, "int": int, "str": str, "bool": bool,
          float: float, int: int, str: str, bool: bool}


def build_section(cls: type[ModelParams], values: Any) -> ModelParams:
    """Build one parameter section from an instance, None, or a mapping.

    All fields must be given (config.yaml supplies them); values are coerced to the
    declared types (YAML turns unquoted ``1e3`` into a string).

    Raises:
        TypeError: for keys that are not fields of ``cls``.
    """
    if isinstance(values, cls):
        # check whether the model is passed as an object to values,
        # if so, simply take the passed model
        return values
    
    if values is None:
        # return default values if no values are passed
        return cls()
    
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise TypeError(
            f"unknown key(s) for {cls.__name__}: {', '.join(sorted(unknown))}. "
            f"Valid keys are: {', '.join(sorted(known))}")
        
    return cls(**{name: _CASTS.get(known[name].type, lambda x: x)(value)
                  for name, value in values.items()})