from dataclasses import dataclass


@dataclass(frozen=True)
class MassDefinition:
    """A spherical-overdensity mass definition such as ``200c`` or ``500m``.

    Parsing ``"200c"`` into an overdensity and a reference density used to be
    repeated -- along with the same validation ``raise`` -- in the mass
    function, the bias, the virial radius and the mass converter.

    Attributes:
        delta: the overdensity, e.g. ``200``.
        reference: ``"c"`` for critical density, ``"m"`` for mean matter density.
    """

    delta: int
    reference: str

    @classmethod
    def parse(cls, text: str) -> "MassDefinition":
        """Parse a definition string like ``"200c"``.

        Raises:
            ValueError: if the string is not an integer followed by ``c`` or ``m``.
        """
        text = str(text).strip()
        reference = text[-1:].lower()
        if reference not in ("c", "m"):
            raise ValueError(
                f"invalid mass definition {text!r}: must end in 'c' (critical) "
                "or 'm' (mean matter), e.g. '200c'"
            )
        try:
            delta = int(text[:-1])
        except ValueError:
            raise ValueError(
                f"invalid mass definition {text!r}: expected an integer "
                f"overdensity before {reference!r}, e.g. '200c'"
            ) from None
        if delta <= 0:
            # Caught here rather than downstream, where it would surface as a
            # division by zero in R_virSO or a cube root of a negative number.
            raise ValueError(
                f"invalid mass definition {text!r}: the overdensity must be "
                f"positive, got {delta}"
            )
        return cls(delta=delta, reference=reference)

    def background_density(self, cosmo, z: float) -> float:
        """Reference density at redshift ``z``, in Msun h^2 / Mpc^3.

        Colossus returns densities per kpc^3; the ``1e9`` converts to per Mpc^3.
        """
        rho = cosmo.rho_c(z=z) if self.reference == "c" else cosmo.rho_m(z=z)
        return rho * 1e9

    def __str__(self) -> str:
        return f"{self.delta}{self.reference}"
