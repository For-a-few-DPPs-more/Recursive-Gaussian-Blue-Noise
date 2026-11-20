"""
Visualisation helpers.

plot                   Scatter plot of a 2-D or 3-D point set.
plot_structure_factor  Log-log plot of the radial structure factor S(k).
"""

from __future__ import annotations

import warnings
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from .structure_factor import structure_factor_and_average, tapered_kmin


def _show_or_return(fig, axes, return_fig: bool):
    plt.tight_layout()
    if return_fig:
        return fig, axes
    plt.show()
    return None


# ---------------------------------------------------------------------
# PLOT POINT SET
# ---------------------------------------------------------------------

def plot(
    points: NDArray,
    auto_zoom: bool = False,
    max_points: int = 30_000,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    figsize: tuple = (8, 8),
    **scatter_kw,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Scatter plot of a 2-D or 3-D point set.

    Parameters
    ----------
    points : ndarray of shape (N, D) or (N, K, D)
        Point coordinates, with D in {2, 3}.
        Arrays of shape (N, K, D) are flattened to (N*K, D) before
        plotting. Dimensions beyond 3 are silently dropped (only
        the first 3 coordinates are displayed).
    auto_zoom : bool, default False
        If True, large point sets are zoomed in so that at most
        ``max_points`` points are rendered, keeping the figure readable.
    max_points : int, default 30_000
        Maximum number of points to render when ``auto_zoom`` is True.
    ax : matplotlib Axes, optional
        Existing axes to draw into. If None, a new figure is created.
        For 3-D data, the axes must have ``projection="3d"``.
    return_fig : bool, default False
        If True, return ``(fig, ax)`` instead of calling ``plt.show()``.
    figsize : tuple, default (8, 8)
        Figure size in inches, forwarded to ``plt.figure()``.
        Ignored when ``ax`` is provided.
    **scatter_kw
        Extra keyword arguments forwarded to ``ax.scatter``
        (e.g. ``color``, ``s``, ``alpha``).

    Returns
    -------
    (fig, ax) if return_fig is True, else None.
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    D = min(pts.shape[-1], 3)
    pts = pts[:, :D]

    if auto_zoom and len(pts) > max_points:
        # keep a random sub-box holding ~max_points points
        unit = (pts - pts.min()) / (pts.max() - pts.min())
        zoom = (max_points / len(pts)) ** (1.0 / D)
        lo = (1 - zoom) * np.random.rand(D)
        pts = pts[((lo <= unit) & (unit <= lo + zoom)).all(axis=1)]

    kw = dict(s=10_000 / len(pts), color="black")
    kw.update(scatter_kw)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if D == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if D == 2:
        ax.scatter(pts[:, 0], pts[:, 1], **kw)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kw)

    ax.set_axis_off()
    return _show_or_return(fig, ax, return_fig)


# ---------------------------------------------------------------------
# STRUCTURE FACTOR
# ---------------------------------------------------------------------

def plot_structure_factor(
    points: NDArray,
    resolution: int = 20_000,
    precision: str = "float32",
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    title: str = None,
    trust_boundary: bool | str = True,
    margin: float = 0.1,
    beta: float | None = None,
    **plot_kw,
) -> tuple[plt.Figure, plt.Axes] | None:
    r"""
    Log-log plot of the radial structure factor S(k).

    The structure factor is estimated from the scattering intensity, i.e. the
    squared modulus of the empirical Fourier transform, evaluated at a set of
    wave vectors from a dual lattice with wave-number up to ``n = 2 * N**(1/D)``.

    The wave vectors are normalised such that k=1 corresponds to the wave-number
    n = N**(1/D) associated to the inter-particle distance.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        Point coordinates in the unit hypercube ``[0, 1)^D``.
    resolution : int, default=20000
        Number of sampled wave vectors. Larger values produce a smoother
        radial curve at increased computational cost. If D >= 4, the resolution
        will be divided by 10 for faster computation because fast fourier is unavailable, so if D >= 4, ask for a "resolution = 200_000" if an effective resolution = 20_000 is needed (20_000 evaluated wave vectors).
    precision : precision used for the structure factor computation.
        float64 is much slower, but might be needed if the true structure
        factor is beyond 10^(-10).
    ax : matplotlib.axes.Axes, optional
        Existing axes on which to draw. If None, a new figure is created.
    return_fig : bool, default=False
        If True, return ``(fig, ax)`` instead of displaying the figure.
    title : str, optional
        Title of the axes. None: a default title depending on trust_boundary.
    trust_boundary : {True, False, "both"}, default=True
        Whether the boundary of the unit hypercube can be trusted.

        - ``True``: the boundary is trusted, no correction is applied.
        The point pattern is assumed to already self compensate for density
        fluctuations near the edges of the unit hypercube, e.g. when the
        points were sampled with periodic boundary conditions.
        - ``False``: the boundary is not trusted, e.g. when the sample is
        only a small window of a larger point pattern on :math:`\mathbb{R}^d`.
        The points are first restricted to the sub-cube
        :math:`[m, 1-m)^d` (``m = margin``), and the estimate on this
        restriction is then debiased with tapering.
        - ``"both"``: compute and plot both estimates.
    margin : float, default=0.1
        Width removed on each side of the unit cube (trust_boundary=False only).
    beta : float or None, default=None
        Kaiser taper parameter (trust_boundary=False only): larger means less
        leakage (~exp(-2 beta)) but coarser resolution at small k. None ->
        automatic (15 when enough points per axis, smaller in high dimension).
    **plot_kw
        Additional keyword arguments forwarded to ``ax.loglog`` for the
        smoothed curve.

    Returns
    -------
    (fig, ax) or None
        Returns ``(fig, ax)`` if ``return_fig=True``; otherwise returns None.

    Notes
    -----
    The wave number is normalized by the characteristic inter-particle
    spacing ``N**(-1/D)``. Thus, ``k=1`` corresponds to the physical wave
    number ``N**(1/D)`` for a unit-volume point set.

    For an order alpha hyperuniform point process, one expects
    ``S(k) ≈ k**alpha`` for k << 1. More generally, suppression of S(k) at
    small k is the characteristic spectral signature of hyperuniformity.

    The plotted scattering intensity is an estimator of the structure factor,
    not the exact structure factor. In particular, finite-window boundary effects can introduce a bias. Use trust_boundary = "both" or False to apply
    boundary effect correction based on tapering.

    Warnings
    --------
    The input points are assumed to lie in the unit hypercube ``[0, 1)^D``.
    This is required for the normalization and is not checked inside the
    estimator.
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    N, D = pts.shape

    if isinstance(trust_boundary, str):
        if trust_boundary != "both":
            raise ValueError("trust_boundary must be True, False or 'both'")
        modes = [True, False]
        both = True
    else:
        modes = [bool(trust_boundary)]
        both = False

    # --- Axes setup ---
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.4, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for trust in modes:
        # --- Compute structure factor and averages ---
        k, S, kgroup, Sgroup = structure_factor_and_average(
            pts, resolution=resolution, precision=precision,
            trust_boundary=trust, margin=margin, beta=beta,
        )
        if not trust:
            # nothing is shown below the resolution limit of the taper
            kmin = tapered_kmin(N, D, margin, beta)
            keep, keep_g = k >= kmin, kgroup >= kmin
            k, S, kgroup, Sgroup = k[keep], S[keep], kgroup[keep_g], Sgroup[keep_g]
            if len(kgroup) < 2:
                warnings.warn("trust_boundary=False: no wave number above the resolution "
                              "limit (too few points in the truncated window); curve not drawn.")
                continue

        # --- Scatter (raw values) ---
        # single mode: neutral grays; 'both': same hue as the smoothed curve,
        # bluish gray (trusted, tab:blue) / orangish gray (tapered, tab:orange)
        if not both:
            c_light, c_dark = "lightgray", "gray"
        elif trust:
            c_light, c_dark = "#b8c4d0", "#6b7f94"
        else:
            c_light, c_dark = "#d8c7b6", "#9c8068"
        Sgroup_interp = np.exp(np.interp(np.log(k), np.log(kgroup), np.log(Sgroup)))
        bigS = S >= 30 * Sgroup_interp
        ax.scatter(k[~bigS], S[~bigS], s=5, color=c_light, alpha=0.6, zorder=2)
        ax.scatter(k[bigS], S[bigS], s=20, color=c_dark, alpha=1.0, zorder=2)

        # --- Smoothed curve ---
        curve_kw = {"color": "tab:blue", "zorder": 3, "label": None,
                    "marker": "o", "markersize": 2, "linewidth": 2}
        curve_kw.update(plot_kw)
        if both:
            curve_kw["label"] = "boundary trusted" if trust else "truncated + taper"
            if not trust:
                curve_kw["color"] = "tab:orange"
                curve_kw["zorder"] = 4
        ax.loglog(kgroup, Sgroup, **curve_kw)

    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    # --- Reference lines and axis limits (only when we own the figure) ---
    if own_fig:
        ax.axvline(1, color="green", linewidth=2, alpha=0.2, zorder=1)
        ax.axhline(1, color="green", linewidth=2, alpha=0.2, zorder=1)
        ax.set_xlim(right=2)
        ax.set_ylim(top=2)
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$S(k)$")

    if both:
        default_title = "Structure factor (log-log): boundary trusted vs truncated + taper"
    elif not modes[0]:
        default_title = "Structure factor (log-log, truncated + Kaiser taper)"
    else:
        default_title = "Structure factor (log-log, scattering intensity)"
    ax.set_title(title or default_title)
    return _show_or_return(fig, ax, return_fig)