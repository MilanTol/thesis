"""Halo density profiles in Fourier space."""

from abc import ABC, abstractmethod

__all__ = ["Profile", "CompositeProfile"]


class Profile(ABC):
    def __init__(self, **kwargs):
        return
    
    @abstractmethod
    def _fourier(self, cosmo, k, M, z, **kwargs):
        """Implementation of the normalised Fourier-space profile."""
        pass
    
    def fourier(self, cosmo, k, M, z, **kwargs):
        """Normalised Fourier transform of the profile.

        Args:
            cosmo: Colossus cosmology object.
            k: wavenumber [h/Mpc].
            M: halo mass [Msun/h].
            z: redshift.
            **kwargs: passed through to the implementation. Composite profiles
                use ``M_parent`` to weight a clump by its host halo; profiles
                that do not need it ignore it.

        Returns:
            The profile in Fourier space, normalised to 1 as ``k -> 0``.
        """
        return self._fourier(cosmo, k, M, z, **kwargs)
       

class CompositeProfile(Profile):
    """A sum of component profiles, each carrying a mass-fraction weight.

    This is how baryons enter the model: a halo is dark matter plus a central
    galaxy plus gas, and each component holds a different fraction of the total
    mass and has a different shape.

    Args:
        *components: the profiles to sum.
        weights: one callable per component, called as
            ``weight(cosmo, M, z, **kwargs)``, giving that component's fraction
            of the halo mass.
        arg_weights: optional per-component callables with the same signature,
            rescaling the *mass argument* passed to that component. This is
            what lets the smooth dark-matter component be evaluated at the
            smooth mass ``(1 - f_clump) * M`` while the stellar and gas
            components stay at the total mass ``M``. Defaults to no rescaling.

    Raises:
        ValueError: if ``weights`` or ``arg_weights`` do not match the number
            of components.
    """

    def __init__(self, *components:Profile, weights, arg_weights=None):
        self._components = [] 
        for component in components: # unpack all components
            self._components.append(component)             
        self._weights = weights 
    
        if arg_weights is None:
            self._arg_weights = [None] * len(self._components)
        else:
            self._arg_weights = arg_weights 
            
                
    def _fourier(self, cosmo, k, M, z, **kwargs):
        total = 0
        for comp, w, aw in zip(self._components, self._weights, self._arg_weights):
            weight = w(cosmo, M, z, **kwargs)
            if aw is None:
                arg_weight = 1
            else:
                arg_weight = aw(cosmo, M, z, **kwargs)
            total += weight * comp.fourier(cosmo, k, arg_weight*M, z, **kwargs)
        return total
    
    
    def __repr__(self):
        names = " + ".join(type(c).__name__ for c in self._components)
        return f"CompositeProfile({names})"


