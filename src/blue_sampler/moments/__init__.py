"""
Moment matching: replace a cell (a cluster of atoms or a polygon) by n equal-weight
points sharing its centroid and central moments up to order p - 1.

Notation follows Lotz & Klatt, "Persistence of asymptotic variance under transport"
(arXiv:2605.22803), Eq. (1.8):

    1/lambda_d(C) int_C x^q dx = 1/n sum_j X_j^q,    |q| in {0, ..., p-1}

q = 0 holds by normalisation, q = 1 fixes the centroid, 2 <= |q| <= p-1 are the
central-moment constraints, solved by Levenberg-Marquardt.
"""
from .fit import fit_points, required_n, moment_orders

__all__ = ["fit_points", "required_n", "moment_orders"]
