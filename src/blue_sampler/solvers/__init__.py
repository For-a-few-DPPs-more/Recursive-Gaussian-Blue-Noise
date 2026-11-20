"""
Point-set samplers.

bruteforce   exact O(N^2) gaussian blue noise (jax, or KeOps)
rgbn         recursive gaussian blue noise (grid or Verlet neighbours)
nufft        spectral optimisation of the structure factor (finufft, jax or KeOps)
tessels      fair recursive splits of the unit square into quadrilaterals (2D)
clusters     fair recursive splits of a point set into clusters (any D), cstit pipeline
kernels      pairwise gaussian forces (bruteforce, rgbn)
geometry     polygons, simplices, random rotations
"""
