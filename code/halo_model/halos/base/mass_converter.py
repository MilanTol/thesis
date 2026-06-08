import numpy as np
from scipy.optimize import brentq

from ...config.config import Config

# converts masses in NFW halos from one definition to another

class NFWMassConverter:
    
    def __init__(self, cfg:Config):
        self.cfg = cfg
    
    
    def _f(self, c:float)->float:
        """
        NFW helper function
        
        Args:
            c (float): concentration
        """
        return np.log(1.0 + c) - c/(1.0 + c)
    
    
    def M_enclosed(self, R):
        return 
    
    
    def _reference_density(self, ref_type:str, z:float) -> float:
        """
        Return the reference density in  M_sun / Mpc^3  for a given type.
    
        Parameters
        ----------
        ref_type : 'c' for critical, 'm' for mean matter
        cosmo    : astropy cosmology object
        z        : redshift
        """
        if ref_type == "c":
            return self.cfg.cosmo.rho_c(z)
        elif ref_type == "m":
            return self.cfg.cosmo.rho_m(z)
        else:
            raise ValueError(f"Unknown reference type '{ref_type}'. Use 'c' or 'm'.")
        
        
    def mass_converter(self, massdef_in:str, massdef_out:str, 
                       M_in:float, c_in:float, z:float):
        """
        Converts the mass of a halo from one definition to another.
        The mass definitions must be of the type <int>c or <int>m.
        (eg. 200m or 500c).
        This also requires the concentration of the halo consistent with
        the inputted definition. 

        Args:
            massdef_in (str): inputted mass definition.
            massdef_out (str): desired output mass definition
            M_in (float): the halo mass for the input definition
            c_in (float): the halo concentration consistent with input definition
            z (float): redshift of the halo. (important for reference densities)

        Returns:
            M_out (float) : Mass of the halo for the desired output definition
        """
        
        # extract reference density and overdensity from inputted mass definition
        reference_density_in = self._reference_density(massdef_in[-1], z)
        Delta_in = int(massdef_in[:-1])
        
        # extract reference density and overdensity from ouputted mass definition
        reference_density_out = self._reference_density(massdef_out[-1], z)
        Delta_out = int(massdef_out[:-1])
        
        # compute the virial radius of the input definition
        R_in = ((3*M_in)/(4*np.pi * Delta_in*reference_density_in))**(1/3) 
        
        # compute the scale radius of the NFW profile (definition independent)
        R_s = R_in / c_in
        
        # compute the constant rho_0 density (definition independent)
        rho_0 = M_in / (4*np.pi * R_s**3 * self._f(c_in))
        
        # now we can solve for the virial radius of the output definition.
        # the virial radius will be at the root of func:
        def func(logx):
            x = np.exp(logx)
            """
            Input x is R / R_s
            """
            term1 = 1/3 * x**3 * Delta_out * reference_density_out
            term2 = rho_0 * (np.log(1 + x) - 1/(1 + 1/x))
            return term1 - term2
        
        # use brents method to find the root:
        logx = brentq(func, np.log(1e-4), np.log(1e4), rtol=1e-5)
        R_out = R_s * np.exp(logx)
        
        # compute the mass for the desired definition:
        M_out = 4*np.pi/3 * R_out**3 * Delta_out * reference_density_out
                
        return M_out
    
    def __call__(self, massdef_in:str, massdef_out:str, 
                M_in:float, c_in:float, z:float):
        """
        Converts the mass of a halo from one definition to another.
        The mass definitions must be of the type <int>c or <int>m.
        (eg. 200m or 500c).
        This also requires the concentration of the halo consistent with
        the inputted definition. 

        Args:
            massdef_in (str): inputted mass definition.
            massdef_out (str): desired output mass definition
            M_in (float): the halo mass for the input definition
            c_in (float): the halo concentration consistent with input definition
            z (float): redshift of the halo. (important for reference densities)

        Returns:
            M_out (float) : Mass of the halo for the desired output definition
        """
        return self.mass_converter(massdef_in, massdef_out, M_in, c_in, z)
        
        

        
    
        
