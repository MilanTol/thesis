R"""
Config class which contains all relevant input parameter values
for runs with this pipeline. Many functions require this object as input.
It also contains further functionality through methods 
such as returning the grid of values of halo mass, as given by its
resolution inputted in the config.
"""
"""Model configuration.

:class:`Config` is a frozen dataclass loaded from a YAML file. Being frozen
matters: several places used to build a variant of a config by deep-copying it
and calling ``setattr``, and ``AngularPower`` mutated ``cfg.z`` inside a loop,
silently changing the config its caller still held. Use :meth:`Config.replace`
instead -- it returns a new config and leaves the original untouched.
"""

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from colossus.cosmology import cosmology

from .mass_definition import MassDefinition


__all__ = ["Config", "MassDefinition", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yaml"

#: Attributes holding lazily built objects, dropped when pickling.
_CACHED_ATTRS = ("_cosmo", "_mass_definition")


@dataclass(frozen=True)
class Config:
    """Every parameter of the model, loaded from ``config.yaml``.

    Build one with :meth:`from_yaml` or :meth:`default`, and derive variants
    with :meth:`replace`.
    """

    # --- matter power spectrum grid ---
    z: float = 0.0
    m_min: float = 1e3        # minimum clump mass [Msun/h]
    N_m: int = 64             # clump-mass integration steps (power of two)
    M_min: float = 1e3        # minimum halo mass [Msun/h]
    M_max: float = 1e16       # maximum halo mass [Msun/h]
    N_M: int = 256            # halo-mass integration steps (power of two)
    k_min: float = 1e-1       # minimum wavenumber [h/Mpc]
    k_max: float = 1e4        # maximum wavenumber [h/Mpc]
    N_k: int = 64             # wavenumber grid points
    massdef: str = "200c"     # spherical-overdensity mass definition

    # --- clump structure ---
    Q: float = 1.33           # clump concentration scaling
    D: float = 1.0            # clump spatial-distribution concentration scaling

    # --- gas profile (Siegel et al. 2025, Table 1) ---
    logM_c: float = 13.6
    theta_ej: float = 6.0
    mu: float = 0.9
    gamma: float = 2.0
    delta: float = 8.0
    logM_gaslimit: float = 10.0

    # --- stellar profile ---
    r_star: float = 0.015     # galaxy scale radius in units of R200c

    # --- stellar-to-halo mass relation ---
    S: float = 1.0            # overall scaling of the stellar fraction

    # --- convergence power spectrum ---
    z_min: float = 0.0
    z_max: float = 2.0
    N_z: int = 6              # redshifts sampled when building P(k, z)
    z_sources: float = 2.0    # source-plane redshift for the lensing kernel
    l_min: float = 1e1
    l_max: float = 1e6
    N_l: int = 256

    # --- angular correlation function ---
    theta_min: float = 1.0    # minimum angular scale for xi+ [arcsec]
    theta_max: float = 1e3    # maximum angular scale for xi+ [arcsec]
    N_theta: int = 10

    # --- halo mass function modification: R(M) = (1 + M0/M)**alpha ---
    logM0: float = 9.0
    alpha: float = 0.0

    # --- clump mass function modification: R(m) = (1 + m0/m)**beta ---
    logm0: float = 9.0
    beta: float = 0.0

    # --- numerics ---
    integration_method: str = "romb"   # "romb" (fixed grid) or "quad" (adaptive)
    epsrel: float = 1e-4               # relative tolerance when using "quad"
    processes: int | None = None       # worker processes; 1 disables multiprocessing

    # --- cosmology ---
    cosmo_params: Mapping[str, Any] = field(default_factory=lambda: {
        "flat": True, "H0": 70, "Om0": 0.2793,
        "Ob0": 0.0463, "sigma8": 0.821, "ns": 0.972,
    })


    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        """Load a config from a YAML file.

        Args:
            path: path to the YAML file. ``None`` loads the default config
                shipped inside the package, so this works from any machine and
                any working directory.
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

        Raises:
            TypeError: if ``values`` holds a key that is not a config field.
        """
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
        
        Args: 
            **overrides: any parameters to be overridden from original.
        """
        return dataclasses.replace(self, **{
            name: _coerce(_field(type(self), name), value)
            for name, value in overrides.items()
        })

    def vary(self, params: str | Iterable[str],
             values: float | Iterable[float]) -> "Config":
        """Return a copy with one or several named parameters set.

        Useful when a sweep produces parallel name/value sequences;
        :meth:`replace` reads better when the names are literals.

        Args:
            params: a parameter name, or a sequence of names.
            values: the matching value, or sequence of values.

        Raises:
            ValueError: if the two sequences have different lengths.
        """
        if isinstance(params, str):
            return self.replace(**{params: values})
        params, values = list(params), list(values)
        if len(params) != len(values):
            raise ValueError(
                f"got {len(params)} parameter name(s) but {len(values)} value(s)")
        return self.replace(**dict(zip(params, values)))

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

    @property
    def M_c(self) -> float:
        """Characteristic mass of the gas profile."""
        return 10.0 ** self.logM_c

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
        """The config as a plain dict, excluding cached derived objects."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

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


