"""Tabulating expensive functions on a grid and interpolating them.

Two classes used to perform the same block: build a log grid, allocate one
array per output, hand ``(i, j)`` pairs to a :class:`multiprocessing.Pool`, then
recover each result's slot with ``np.where(grid == value)[0][0]``.  That lookup
was both O(N) per point and dependent on exact float equality surviving a round
trip through ``exp``/``log``.  :func:`tabulate_grid` carries the indices
through instead.
Then interpolate logarithmically over the 2d array.
"""

from __future__ import annotations

import os
from multiprocessing import Pool, cpu_count
from typing import Callable, Sequence

import numpy as np
from scipy import interpolate

__all__ = ["tabulate_grid", "LogInterpolator2D"]


def tabulate_grid(worker: Callable[[tuple], tuple], 
                  grids: Sequence[np.ndarray],
                  extra: tuple = (), 
                  n_outputs: int = 1,
                  processes: int | None = None) -> list[np.ndarray]:
    """Evaluate ``worker`` at every point of the outer product of ``grids``.

    Args:
        worker: a module-level (picklable) function called as
            ``worker((*extra, *point))`` and returning either a scalar or a
            tuple of ``n_outputs`` scalars.
        grids: one 1-D array of coordinates per axis.
        extra: leading arguments prepended to every call, e.g. the model object.
        n_outputs: how many values ``worker`` returns per point.
        processes: worker processes; ``1`` runs serially in this process.

    Returns:
        A list of ``n_outputs`` arrays, each shaped like the grid.
    """
    shape = tuple(len(g) for g in grids)
    extra = tuple(extra)
    points = [tuple(values) for values in _cartesian_product(grids)]

    if processes is None:
        env = os.environ.get("HALO_MODEL_PROCESSES")
        processes = int(env) if env else cpu_count()
        processes = max(1, int(processes))
        
    if processes == 1 or len(points) < 2:
        # extra and point are tuples, but these should be combined into 1
        # for this use + operator
        results = [worker(extra + point) for point in points]
    else:
        # `extra` usually holds the whole model, which can be a megabyte or
        # more. Send it once per worker process via the initializer rather
        # than once per grid point. With a default 64x256 grid the latter
        # meant streaming gigabytes through the pool.
        with Pool(processes=processes, initializer=_init_worker,
                  initargs=(worker, extra)) as pool:
            results = pool.map(_run_worker, points)

    outputs = [np.zeros(shape) for _ in range(n_outputs)]
    for flat_index, result in enumerate(results):
        index = np.unravel_index(flat_index, shape)
        values = result if n_outputs > 1 else (result,)
        for output, value in zip(outputs, values):
            output[index] = value
    return outputs


#: Per-process state for the pool, populated by :func:`_init_worker`.
_WORKER_STATE: dict = {}


def _init_worker(worker, extra) -> None:
    """Store the worker and its fixed leading arguments in a pool process."""
    _WORKER_STATE["worker"] = worker
    _WORKER_STATE["extra"] = extra


def _run_worker(point):
    """Call the worker stored by :func:`_init_worker` at one grid point."""
    return _WORKER_STATE["worker"](_WORKER_STATE["extra"] + point)


def _cartesian_product(grids: Sequence[np.ndarray]):
    """Yield grid points in C order, matching :func:`numpy.unravel_index`."""
    if not grids:
        yield ()
        return
    head, *tail = grids
    for value in head:
        for rest in _cartesian_product(tail):
            yield (value,) + rest


class LogInterpolator2D:
    """A function of two variables tabulated on a grid, interpolated in log space.

    Args:
        x_grid: first-axis coordinates, in *linear* units.
        y_grid: second-axis coordinates, in *linear* units.
        values: array of shape ``(len(x_grid), len(y_grid))``.
        log_x: interpolate against ``log(x)`` rather than ``x``.
        log_y: interpolate against ``log(y)`` rather than ``y``.
        clamp_above_x: return 0 beyond the last ``x_grid`` point instead of
            extrapolating. The axes are logarithmic but the *values* are not,
            so a steeply falling quantity extrapolates linearly straight
            through zero and goes negative just past the grid edge. Set this
            for a quantity known to decay to zero, such as the clump integrals
            above ``k_max``.

    Extrapolation is otherwise enabled (``fill_value=None``), matching the
    previous behaviour: the model is routinely evaluated slightly outside the
    tabulated mass range.
    """

    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, values: np.ndarray,
                 *, log_x: bool = True, log_y: bool = True,
                 clamp_above_x: bool = False):
        self._log_x = log_x
        self._log_y = log_y
        self._clamp_above_x = clamp_above_x
        self._x_max = float(np.max(x_grid))
        axes = (np.log(x_grid) if log_x else np.asarray(x_grid),
                np.log(y_grid) if log_y else np.asarray(y_grid))
        self._interp = interpolate.RegularGridInterpolator(
            axes, values, bounds_error=False, fill_value=None)

    def __call__(self, x, y):
        """Interpolated value at ``(x, y)``, given in linear units."""
        value = self._interp((np.log(x) if self._log_x else x,
                              np.log(y) if self._log_y else y))
        if self._clamp_above_x:
            # Strictly above, so a point sitting exactly on the grid edge is
            # still interpolated normally.
            value = np.where(np.asarray(x) > self._x_max, 0.0, value)
        return value
