r"""One place where integrals over mass are performed.

Every mass integral in the halo model has the same shape: an integrand
:math:`g(M)` integrated as :math:`\int g(M)\,\mathrm{d}M` over a range spanning
many decades, which is done in :math:`\ln M` for accuracy.  Previously each of
the ~12 sites that needed one open-coded the log grid, the ``dx``, the list
comprehension, the Romberg call and the ``* M`` Jacobian factor.  They all call
:class:`MassIntegral` instead.

The Jacobian is applied *here*, so an integrand written against this module is
the plain :math:`g(M)` of the equation it implements -- no stray ``* M`` at the
end of every ``return``.
"""

from typing import Callable
import numpy as np
from scipy import integrate


class MassIntegral:
    r"""Integrates ``integrand(M) dM`` on a logarithmic mass grid.

    The substitution :math:`u = \ln M` turns the integral into
    :math:`\int g(M)\,M\,\mathrm{d}u`; the :math:`M` Jacobian is supplied by
    this class.

    This class uses Romberg (romb) integration on a fixed grid of ``n_steps`` intervals
    (``n_steps`` must be a power of two).  Deterministic cost, which is what
    makes the model cheap enough to interpolate over a (k, M) grid.

    Args:
        n_steps: number of intervals for romb (in mass space).  Rounded up to the next
            power of two, since Romberg requires ``2**k + 1`` samples.

    Example:
        >>> integrate_mass = MassIntegral(n_steps=64)
        >>> integrate_mass(lambda M: 1.0 / M, 1e3, 1e16)   # = ln(1e16/1e3)
        29.9336...
    """
    
    def __init__(self, n_steps: int):
        self.n_steps = _next_power_of_two(n_steps)
        
    
    def __call__(self, integrand: Callable[[float], float],
                 m_low: float, m_high: float) -> float:
        """Integrate ``integrand(M) dM`` from ``m_low`` to ``m_high``.

        Args:
            integrand: function of mass ``M``, *without* the ``dlnM -> dM``
                Jacobian, this method multiplies by ``M`` for you.
            m_low: lower mass limit (must be > 0).
            m_high: upper mass limit.

        Returns:
            The value of the integral.  Zero if ``m_high <= m_low``, which
            happens naturally for clump integrals when the parent halo sits at
            the bottom of the mass range.
        """        
        if m_high <= m_low:
            return 0
        
        def log_integrand(ln_m: float) -> float:
            m = np.exp(ln_m)
            jacobian = m # <- includes the jacobian dlnm -> dm
            return integrand(m) * jacobian
        
        ln_low, ln_high = np.log(m_low), np.log(m_high)

        ln_grid = np.linspace(ln_low, ln_high, self.n_steps + 1)
        samples = [log_integrand(ln_m) for ln_m in ln_grid]
        return integrate.romb(samples, ln_grid[1] - ln_grid[0])
    
    
    def __repr__(self) -> str:
        if self.method == "quad":
            return f"MassIntegral(method='quad', epsrel={self.epsrel:g})"
        return f"MassIntegral(method='romb', n_steps={self.n_steps})"
   
        
        
def _next_power_of_two(n: int) -> int:
    """Smallest power of two >= ``n`` (Romberg needs ``2**k + 1`` samples)."""
    if n < 1:
        raise ValueError(f"n_steps must be positive, got {n}")
    return 1 << (int(n) - 1).bit_length()

