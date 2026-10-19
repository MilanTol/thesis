R"""
Config class which contains all relevant input parameter values
for runs with this pipeline. Many functions require this object as input.
It also contains further functionality through methods 
such as returning the grid of values of halo mass, as given by its
resolution inputted in the config.

Model choices (e.g. which gas profile) live in ``models``; each model's own
parameters live in ``params[component][model]`` and are read through
``cfg.p(component)``, which returns the section of the *selected* model.
"""

import copy
import dataclasses
import functools
import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from colossus.cosmology import cosmology

from .mass_definition import MassDefinition
from .params import COMPONENTS, ModelParams, build_section


__all__ = ["Config", "MassDefinition", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yaml"

#: Attributes holding lazily built objects, dropped when pickling.
_CACHED_ATTRS = ("_cosmo", "_mass_definition")

#: Flat gas-profile keys from before model selection existed. Old YAML files
#: using them are moved into ``params.gas_profile.siegel2025`` with a warning.
_LEGACY_SIEGEL_KEYS = ("logM_c", "theta_ej", "mu", "gamma", "delta")


@dataclass(frozen=True)
class Config:
    """Every parameter of the model, loaded from ``config.yaml``.

    No default values live in this class: the single source of defaults is the
    ``config.yaml`` next to this file. Build one with :meth:`default` or
    :meth:`from_yaml` (a partial file is fine, missing values come from the
    packaged ``config.yaml``), and derive variants with :meth:`replace`.
    """

    # --- matter power spectrum grid ---
    z: float
    m_min: float              # minimum clump mass [Msun/h]
    N_m: int                  # clump-mass integration steps (power of two)
    M_min: float              # minimum halo mass [Msun/h]
    M_max: float              # maximum halo mass [Msun/h]
    N_M: int                  # halo-mass integration steps (power of two)
    k_min: float              # minimum wavenumber [h/Mpc]
    k_max: float              # maximum wavenumber [h/Mpc]
    N_k: int                  # wavenumber grid points
    massdef: str              # spherical-overdensity mass definition

    # --- model selection ---
    models: Mapping[str, str]                       # component -> selected model name
    params: Mapping[str, Mapping[str, ModelParams]] # component -> model -> parameters

    # --- clump structure ---
    Q: float                  # clump concentration scaling
    D: float                  # clump spatial-distribution concentration scaling

    # --- gas profile (shared by all gas models) ---
    logM_gaslimit: float

    # --- stellar profile ---
    r_star: float             # galaxy scale radius in units of R200c

    # --- stellar-to-halo mass relation ---
    S: float                  # overall scaling of the stellar fraction

    # --- convergence power spectrum ---
    z_min: float
    z_max: float
    N_z: int                  # redshifts sampled when building P(k, z)
    z_sources: float          # source-plane redshift for the lensing kernel
    l_min: float
    l_max: float
    N_l: int

    # --- angular correlation function ---
    theta_min: float          # minimum angular scale for xi+ [arcsec]
    theta_max: float          # maximum angular scale for xi+ [arcsec]
    N_theta: int

    # --- halo mass function modification: R(M) = (1 + M0/M)**alpha ---
    logM0: float
    alpha: float

    # --- clump mass function modification: R(m) = (1 + m0/m)**beta ---
    logm0: float
    beta: float

    # --- numerics ---
    integration_method: str   # "romb" (fixed grid) or "quad" (adaptive)
    epsrel: float             # relative tolerance when using "quad"
    processes: int | None     # worker processes; 1 disables multiprocessing

    # --- cosmology ---
    cosmo_params: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the model selection and build every parameter section."""
        models = dict(self.models)
        missing = set(COMPONENTS) - set(models)
        if missing:
            raise TypeError(
                f"no model selected for component(s): {', '.join(sorted(missing))}")
        for component, name in models.items():
            if component not in COMPONENTS:
                raise ValueError(
                    f"unknown model component {component!r}. "
                    f"Valid: {', '.join(sorted(COMPONENTS))}")
            if name not in COMPONENTS[component]:
                raise ValueError(
                    f"unknown {component} model {name!r}. "
                    f"Valid: {', '.join(sorted(COMPONENTS[component]))}")

        unknown = set(self.params) - set(COMPONENTS)
        if unknown:
            raise TypeError(
                f"unknown component(s) in params: {', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(sorted(COMPONENTS))}")

        built: dict[str, dict[str, ModelParams]] = {}
        for component, table in COMPONENTS.items():
            raw = dict(self.params.get(component) or {})
            unknown = set(raw) - set(table)
            if unknown:
                raise TypeError(
                    f"unknown {component} model(s) in params: "
                    f"{', '.join(sorted(unknown))}. Valid: {', '.join(sorted(table))}")
            built[component] = {name: build_section(cls, raw.get(name))
                                for name, cls in table.items()}

        object.__setattr__(self, "models", models)
        object.__setattr__(self, "params", built)

        # only the selected model of each component is validated
        for component, name in models.items():
            built[component][name].validate()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        """Load a config from a YAML file.

        Args:
            path: path to the YAML file. ``None`` loads the default config
                shipped inside the package, so this works from any machine and
                any working directory. The file may be partial: anything it
                leaves out is taken from the packaged ``config.yaml``.
            **overrides: fields to override after loading.

        Raises:
            FileNotFoundError: if ``path`` is given but does not exist.
            TypeError: if the file contains an unknown key.
        """
        path = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path) as stream:
            values = yaml.safe_load(stream.read()) or {}
        return cls.from_dict({**values, **overrides})

    @classmethod
    def default(cls, **overrides: Any) -> "Config":
        """The config shipped with the package, with optional overrides."""
        return cls.from_yaml(None, **overrides)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Config":
        """Build a config from a mapping, coercing values to the declared types.

        ``values`` is layered over the packaged ``config.yaml`` (nested
        mappings such as ``params`` and ``cosmo_params`` are merged key by key),
        so it only needs to hold what differs from the defaults.

        Raises:
            TypeError: if ``values`` holds a key that is not a config field.
        """
        values = _deep_merge(_packaged_defaults(),
                             _migrate_legacy_keys(dict(values)))
        known = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(values) - set(known)
        if unknown:
            raise TypeError(
                f"unknown config key(s): {', '.join(sorted(unknown))}. "
                f"Valid keys are: {', '.join(sorted(known))}"
            )
        return cls(**{name: _coerce(known[name], value)
                      for name, value in values.items()})

    def replace(self, **overrides: Any) -> "Config":
        """Return a copy of this config with some fields changed.

        This is the supported way to vary a parameter: it never mutates the
        receiver, so one config can be shared safely across a parameter sweep
        or a multiprocessing pool.

        Three kinds of keys are understood:

        * a plain field name (``z=0.5``),
        * a component name, which switches the selected model
          (``gas_profile="debackere2022"``),
        * ``"component.parameter"``, which sets a parameter of the model
          *selected at that point* (``**{"gas_profile.gamma": 1.2}``).
          Switching the model and setting its parameters can be combined in
          one call; the switch is applied first.
        
        Args: 
            **overrides: any parameters to be overridden from original.
            
        Example:
            >>> base = Config(z=0.0)
            >>> c1 = base.replace(z=0.5)
            >>> c2 = base.replace(gas_profile="debackere2022", **{"gas_profile.gamma": 1.2})
            >>> base.z   # base is never mutated
            0.0
            >>> c1.z
            0.5
        """
        plain: dict[str, Any] = {}
        selectors: dict[str, str] = {}
        dotted: dict[str, Any] = {}
        for name, value in overrides.items():
            if "." in name:
                dotted[name] = value
            elif name in COMPONENTS:
                selectors[name] = value
            else:
                plain[name] = _coerce(_field(type(self), name), value)

        if selectors:
            plain["models"] = {**plain.get("models", self.models), **selectors}
        cfg = dataclasses.replace(self, **plain) if plain else self

        if dotted:
            params = {c: dict(t) for c, t in cfg.params.items()}
            for name, value in dotted.items():
                component, key = name.split(".", 1)
                if component not in COMPONENTS:
                    raise TypeError(
                        f"unknown model component {component!r} in {name!r}. "
                        f"Valid: {', '.join(sorted(COMPONENTS))}")
                selected = cfg.models[component]
                section = params[component][selected]
                params[component][selected] = build_section(
                    type(section), {**dataclasses.asdict(section), key: value})
            cfg = dataclasses.replace(cfg, params=params)
        return cfg

    def vary(self, params: str | Iterable[str],
             values: float | Iterable[float]) -> "Config":
        """Return a copy with one or several named parameters set.

        Useful when a sweep produces parallel name/value sequences;
        :meth:`replace` reads better when the names are literals. Accepts the
        same keys as :meth:`replace`, including ``"component.parameter"``.

        Args:
            params: a parameter name, or a sequence of names.
            values: the matching value, or sequence of values.

        Raises:
            ValueError: if the two sequences have different lengths.
        
        Example:
            One parameter, one value:
            >>> c = base.vary("sigma8", 0.85)

            Several parameters, parallel sequences:
            >>> c = base.vary(["omega_m", "sigma8"], [0.30, 0.82]) 

            A parameter of the selected gas model:
            >>> c = base.vary("gas_profile.gamma", 1.2)
        """
        if isinstance(params, str):
            return self.replace(**{params: values})
        params, values = list(params), list(values)
        if len(params) != len(values):
            raise ValueError(
                f"got {len(params)} parameter name(s) but {len(values)} value(s)")
        return self.replace(**dict(zip(params, values)))

    # ------------------------------------------------------------------
    # model selection
    # ------------------------------------------------------------------

    def model(self, component: str) -> str:
        """Name of the model selected for ``component`` (e.g. ``"siegel2025"``)."""
        return self.models[component]

    def p(self, component: str) -> ModelParams:
        """Parameter section of the model selected for ``component``.

        Example:
            >>> cfg.p("gas_profile").gamma
        """
        return self.params[component][self.models[component]]

    def with_model(self, component: str, name: str) -> "Config":
        """Return a copy with a different model selected for ``component``."""
        return self.replace(**{component: name})

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------

    @property
    def cosmology_name(self) -> str:
        """Name this config's cosmology is registered under in Colossus.

        Derived from the parameters themselves, so two configs with the same
        cosmology share one registration and a name is stable across processes.
        """
        digest = hashlib.md5(
            repr(sorted(dict(self.cosmo_params).items())).encode()).hexdigest()[:12]
        return f"halo_model_{digest}"

    @property
    def cosmo(self):
        """The Colossus cosmology built from :attr:`cosmo_params`.

        Accessing this also makes it Colossus's *current* cosmology -- see
        :meth:`activate`.
        """
        return self.activate()

    def activate(self):
        """Make this config's cosmology the current Colossus cosmology.

        Colossus keeps a single module-global "current" cosmology, and several
        of the functions this package calls -- ``massFunction``, ``haloBias``,
        ``matterPowerSpectrum`` -- read it rather than taking a cosmology
        argument. Two things follow:

        * A freshly spawned worker process has no current cosmology at all, so
          anything that unpickles a config must re-establish it. That is why
          the mass integrals call this before they run.
        * Interleaving two models built from different cosmologies would
          otherwise let one silently pick up the other's parameters.

        Cheap to call repeatedly: it returns immediately once this config's
        cosmology is already current.

        Returns:
            The Colossus cosmology object.
        """
        cached = self.__dict__.get("_cosmo")
        try:
            if cached is not None and cosmology.getCurrent() is cached:
                return cached
        except Exception:

            pass   # no current cosmology set yet (a fresh worker process)

        name = self.cosmology_name
        if name not in cosmology.cosmologies:
            cosmology.addCosmology(name, **dict(self.cosmo_params))
        cached = cosmology.setCosmology(name)
        object.__setattr__(self, "_cosmo", cached)
        return cached

    @property
    def mass_definition(self) -> MassDefinition:
        """:attr:`massdef` parsed into an overdensity and a reference density."""
        cached = self.__dict__.get("_mass_definition")
        if cached is None:
            cached = MassDefinition.parse(self.massdef)
            object.__setattr__(self, "_mass_definition", cached)
        return cached

    @property
    def f_b(self) -> float:
        """Cosmic baryon fraction Omega_b / Omega_m, derived from the cosmology."""
        return self.cosmo_params["Ob0"] / self.cosmo_params["Om0"]

    @property
    def M0(self) -> float:
        """Halo mass-function pivot mass, ``10**logM0``."""
        return 10.0 ** self.logM0
    @property
    def m0(self) -> float:
        """Clump mass-function pivot mass, ``10**logm0``."""
        return 10.0 ** self.logm0

    @property
    def M_gaslimit(self) -> float:
        """Halo mass below which the gas profile is held fixed."""
        return 10.0 ** self.logM_gaslimit

    def k_grid(self, n: int | None = None) -> np.ndarray:
        """Logarithmically spaced wavenumber grid, ``k_min`` to ``k_max``."""
        return np.geomspace(self.k_min, self.k_max, n or self.N_k)

    def M_grid(self, n: int | None = None) -> np.ndarray:
        """Logarithmically spaced halo-mass grid, ``M_min`` to ``M_max``."""
        return np.geomspace(self.M_min, self.M_max, n or self.N_M)

    def z_grid(self, descending: bool = False) -> np.ndarray:
        """Redshift grid from ``z_min`` to ``z_max``.

        Args:
            descending: return high redshift first, so the corresponding scale
                factors increase monotonically (required by ``pyccl``).
        """
        grid = np.linspace(self.z_min, self.z_max, self.N_z)
        return grid[::-1] if descending else grid

    def ell_grid(self) -> np.ndarray:
        """Multipole grid used for the convergence spectrum."""
        return np.geomspace(self.l_min, self.l_max, self.N_l + 1)

    def theta_grid_deg(self) -> np.ndarray:
        """Angular-scale grid in degrees (the config stores arcseconds)."""
        return np.geomspace(self.theta_min / 3600, self.theta_max / 3600,
                            self.N_theta)


    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The config as a plain dict, excluding cached derived objects.

        Parameter sections are expanded to plain dicts so the result can be
        dumped to YAML. All models' sections are included, not just the
        selected ones, so a saved config reproduces the run exactly.
        """
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        out["models"] = dict(self.models)
        out["params"] = {
            component: {name: dataclasses.asdict(section)
                        for name, section in table.items()}
            for component, table in self.params.items()
        }
        out["cosmo_params"] = dict(self.cosmo_params)
        return out

    def save(self, path: str | Path) -> Path:
        """Write this config to YAML.

        Args:
            path: a directory (``config.yaml`` is written inside it), or a
                filename ending in ``.yaml`` / ``.yml``.

        Returns:
            The path actually written.
        """
        path = Path(path).expanduser()
        if path.suffix.lower() not in (".yaml", ".yml"):
            path = path / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)
        return path

    def __getstate__(self) -> dict[str, Any]:
        """Drop cached objects so the config pickles cleanly to worker processes."""
        return {k: v for k, v in self.__dict__.items() if k not in _CACHED_ATTRS}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


def config_modifier(cfg: Config, param: str | Iterable[str],
                    val: float | Iterable[float]) -> Config:
    """Deprecated alias for :meth:`Config.vary`.

    Kept so existing notebooks keep working. New code should call
    ``cfg.vary(param, val)`` or ``cfg.replace(**overrides)``.
    """
    return cfg.vary(param, val)


@functools.lru_cache(maxsize=1)
def _read_packaged_defaults() -> dict[str, Any]:
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"packaged default config not found: {DEFAULT_CONFIG_PATH}")
    with open(DEFAULT_CONFIG_PATH) as stream:
        return yaml.safe_load(stream.read()) or {}


def _packaged_defaults() -> dict[str, Any]:
    """The contents of the packaged ``config.yaml`` (a fresh copy each call)."""
    return copy.deepcopy(_read_packaged_defaults())


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``override`` layered over ``base``; nested mappings merge key by key."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _migrate_legacy_keys(values: dict[str, Any]) -> dict[str, Any]:
    """Move old flat Siegel keys (logM_c, theta_ej, ...) into the params section."""
    legacy = {k: values.pop(k) for k in _LEGACY_SIEGEL_KEYS if k in values}
    if not legacy:
        return values
    warnings.warn(
        f"flat gas-profile keys ({', '.join(legacy)}) are deprecated; put them "
        f"under params.gas_profile.siegel2025 in the config",
        DeprecationWarning, stacklevel=3)
    params = {c: dict(t or {}) for c, t in (values.get("params") or {}).items()}
    section = params.setdefault("gas_profile", {}).setdefault("siegel2025", {})
    for key, value in legacy.items():
        section.setdefault(key, value)   # explicit new-style values win
    values["params"] = params
    return values


def _field(cls, name: str) -> dataclasses.Field:
    """Look up a dataclass field by name.

    Raises:
        TypeError: if no such field exists, listing the valid names.
    """
    for candidate in dataclasses.fields(cls):
        if candidate.name == name:
            return candidate
    valid = ", ".join(sorted(f.name for f in dataclasses.fields(cls)))
    raise TypeError(f"unknown config key {name!r}. Valid keys are: {valid}")


def _coerce(spec: dataclasses.Field, value: Any) -> Any:
    """Coerce a YAML value to the field's declared type.

    YAML yields the *string* ``"1e3"`` for an unquoted float in exponent
    notation.
    """
    if value is None:
        return None
    if spec.type in ("float", float):
        return float(value)
    if spec.type in ("int", int):
        return int(value)
    if spec.type in ("str", str):
        return str(value)
    return value