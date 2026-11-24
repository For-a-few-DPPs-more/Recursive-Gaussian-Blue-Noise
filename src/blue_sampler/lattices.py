from structure_factor_v2 import curled_lattice, jittered_lattice, korobov_lattice
from numpy.typing import NDArray

def sobol(N: int = 2 ** 15, D: int = 2, seed: int | None = None) -> NDArray:
    """N points of a scrambled Sobol sequence (best with N a power of 2, SciPy warns otherwise)."""
    from scipy.stats import qmc
    return qmc.Sobol(d=D, seed=seed).random(N)