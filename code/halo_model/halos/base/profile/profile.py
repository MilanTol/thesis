from abc import ABC, abstractmethod

class Profile(ABC):
    @abstractmethod
    def _fourier(self, cosmo, k, M, z, **kwargs):
        """
        Docstring for fourier
        
        :param cosmo: cosmology object
        :param k: wavenumber k (h/Mpc)
        :param M: halo mass
        :param z: redshift 
        """
        pass

    def fourier(self, cosmo, k, M, z, **kwargs):
        """
        Docstring for fourier
        
        :param cosmo: cosmology object
        :param k: wavenumber k (h/Mpc)
        :param M: halo mass
        :param z: redshift 
        """
        return self._fourier(cosmo, k, M, z, **kwargs)
       

class CompositeProfile(Profile):
    """A profile composed of multiple component profiles whose Fourier
    transforms are summed."""

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


