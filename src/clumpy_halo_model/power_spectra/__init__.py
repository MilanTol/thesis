"""Matter, angular and correlation-function spectra.

Only the matter power spectrum is re-exported here. ``param_effect``,
``angular_power``, ``del_xi`` and ``correlation`` are imported directly:

- the first three build models and so would import :mod:`halo_model.models`,
  which imports this package -- a cycle;
- ``correlation`` pulls in the optional ``pyccl`` / ``treecorr`` stack.
"""

from .matter_power import COMPONENTS, MatterPower
from .Pm_computer import Pm_computer

__all__ = ["MatterPower", "COMPONENTS", "Pm_computer"]