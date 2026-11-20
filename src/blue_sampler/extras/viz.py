"""
Visualisation helpers (extras).

plot_tessels           Display a STIT tessellation across recursion depths.
plot_clusters          Display a cluster partition across recursion depths.
plot_polygons          Display an arbitrary collection of polygons.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from ..solvers.geometry import polygon_area
from ..solvers import clusters as _clusters, tessels as _tessels


def _show_or_return(fig, axes, return_fig: bool):
    plt.tight_layout()
    if return_fig:
        return fig, axes
    plt.show()
    return None

# ---------------------------------------------------------------------
# RECURSIVE PARTITIONS (tessels, clusters)
# ---------------------------------------------------------------------

def _plot_levels(items, merge, draw, max_panels, axes, return_fig, prepare=lambda items: items):
    """One panel per recursion depth: draw the finest level, merge siblings, repeat."""
    depth = int(np.log2(len(items)))
    n_plots = min(depth, max_panels)
    ncols = min(3, n_plots) if n_plots > 0 else 1
    nrows = int(np.ceil(n_plots / ncols))
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    else:
        fig = np.ravel(axes)[0].get_figure()
    axes = np.ravel(axes)

    while depth > max_panels - 1:
        items = merge(items)
        depth -= 1
    items = prepare(items)
    for k in reversed(range(n_plots)):
        draw(axes[k], np.random.permutation(items))
        axes[k].set_aspect("equal")
        axes[k].axis("off")
        if k > 0:
            items = merge(items)
    return _show_or_return(fig, axes, return_fig)


def show_polygons(ax: plt.Axes, polygons: NDArray) -> plt.Axes:
    """Draw outlined, lightly filled polygons on ``ax``."""
    for poly in polygons:
        loop = np.vstack([poly, poly[0]])
        ax.plot(loop[:, 0], loop[:, 1], "-o", ms=3)
        ax.fill(loop[:, 0], loop[:, 1], alpha=0.25)
    return ax


def show_clusters(ax: plt.Axes, clusters: NDArray) -> plt.Axes:
    """Scatter-plot clusters (N, k, D>=2) on ``ax``, one colour per cluster (depth-sorted in 3D)."""
    n_clusters, k, D = clusters.shape
    cmap = plt.get_cmap("tab20" if n_clusters > 10 else "tab10")
    pts = clusters.reshape(-1, D)
    order = np.argsort(pts[:, 2]) if D == 3 else np.arange(len(pts))
    colors = cmap(np.repeat(np.arange(n_clusters), k) % cmap.N)
    ax.scatter(pts[order, 0], pts[order, 1], s=4, color=colors[order])
    return ax


def plot_tessels(tessels: NDArray, axes=None, return_fig: bool = False):
    """
    Display a fair tessellation across its recursion depths (up to 12 panels).

    Parameters
    ----------
    tessels : ndarray of shape (N, 4, 2)
        Output of `sample_tessels`; N must be a power of 2.
    axes : array of matplotlib Axes, optional
        Axes to draw into (one per depth). A new figure is created if None.
    return_fig : bool, default False
        Return ``(fig, axes)`` instead of calling ``plt.show()``.

    Examples
    --------
    >>> blue.plot_tessels(blue.sample_tessels(N=1024))
    """
    return _plot_levels(tessels, _tessels.merge_siblings, show_polygons, 12, axes, return_fig)


_ISOMETRIC = np.array([
    [1, 1 / np.sqrt(3), -np.sqrt(2) / np.sqrt(3)],
    [-1, 1 / np.sqrt(3), -np.sqrt(2) / np.sqrt(3)],
    [0, 2 / np.sqrt(3), np.sqrt(2) / np.sqrt(3)],
])


def plot_clusters(clusters: NDArray, axes=None, return_fig: bool = False):
    """
    Display a cluster partition across its recursion depths (up to 9 panels).

    Parameters
    ----------
    clusters : ndarray of shape (N, k, D)
        Output of `sample_clusters`; N must be a power of 2. For D >= 3 the first 3
        coordinates are shown through an isometric projection. At most 16 atoms per
        cluster are drawn.
    axes : array of matplotlib Axes, optional
        Axes to draw into (one per depth). A new figure is created if None.
    return_fig : bool, default False
        Return ``(fig, axes)`` instead of calling ``plt.show()``.

    Examples
    --------
    >>> blue.plot_clusters(blue.sample_clusters(N=512, D=2))
    """
    def prepare(cl):
        cl = np.random.permutation(cl.transpose(1, 0, 2)).transpose(1, 0, 2)[:, :16, :3]
        return np.einsum("ijk,kl->ijl", cl, _ISOMETRIC) if cl.shape[-1] == 3 else cl

    return _plot_levels(clusters, _clusters.merge_siblings, show_clusters, 9, axes, return_fig, prepare)


def plot_polygons(
    polygons: NDArray,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    color: str = "auto",
    linewidth: float = 1.2,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Visualise a collection of polygons with colour-coded faces.

    Polygon opacity is scaled by inverse area squared, so tiny polygons
    (e.g. deep in a Pinwheel tiling) remain visible while large ones are
    rendered with full opacity.

    Parameters
    ----------
    polygons : ndarray of shape (N, M, 2)
        N polygons, each defined by M vertices in 2D.
        Typical inputs:

        - Output of `pinwheel_transform` — triangles of shape (N, 3, 2).
        - Output of `sample_tessels` — quadrilaterals of shape (N, 4, 2).
    ax : matplotlib Axes, optional
        Existing axes to draw into. If None, a new figure is created.
    return_fig : bool, default False
        If True, return ``(fig, ax)`` instead of calling ``plt.show()``.

    Returns
    -------
    (fig, ax) if return_fig is True, else None.

    Examples
    --------
    >>> pw = blue.pinwheel_transform(blue.pinwheel_base(), depth=4)
    >>> blue.plot_polygons(pw)
    """
    if len(polygons) >= 30_000:
        linewidth = 0
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    else:
        fig = ax.get_figure()

    n = len(polygons)
    if color == "auto":
        cmap = plt.cm.YlGnBu
        color = cmap(0.2 + 0.8 * (np.random.permutation(np.arange(n)) / n))

    inv_area2 = 1 / polygon_area(polygons) ** 2
    alpha = (inv_area2 / inv_area2.mean() + 0.01).clip(max=1.0)   # small polygons stay visible

    collection = PolyCollection(
        polygons,
        facecolors=color,
        edgecolors="none" if linewidth <= 0.01 else "#FFFFFF",
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.axis("off")
    return _show_or_return(fig, ax, return_fig)
