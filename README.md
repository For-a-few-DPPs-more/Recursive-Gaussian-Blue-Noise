[![PyPI](https://img.shields.io/pypi/v/blue-sampler.svg)](https://pypi.org/project/blue-sampler/)
[![Docs](https://readthedocs.org/projects/blue-sampler/badge/?version=latest)](https://blue-sampler.readthedocs.io)
[![GitHub](https://img.shields.io/badge/source-GitHub-black?logo=github)](https://github.com/For-a-few-DPPs-more/rgbn)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/For-a-few-DPPs-more/rgbn/blob/main/examples.ipynb)


# blue-sampler

**Generate large stealthy point patterns** on the unit torus $[0, 1)^D$. 📐

Stealthy point patterns exhibit vanishing density fluctuations at low frequencies, making them particularly suited for **Monte Carlo** integration, **image stippling**, and any application requiring well-distributed, low-discrepancy points. 

The main blue-noise samplers (**RGBN** and **NUFFT**) offer **linear** complexity in the number of points.  
They can generate e.g. 1 million 2D (resp 3D) points in under 10 minutes (resp 30 minutes) on a standard CPU, and up to 30× faster on GPU (30s resp 1 minute). 

> **Note on adaptative sampling**: The sampling methods implemented here support adaptive sampling from a target distribution. This feature is still experimental beyond 2d distributions
> **Note on structure factor computation**: The structure factor is estimated from the standard `scattering intensity` and assumes that the point domain is the unit hypercube. By default, the computation uses float32 precision, which limits the reliable estimation of very small structure factors (≲ 1e-10). For ultra-stealthy point patterns with S(k) ≲ 1e-10, enable 64-bit precision in JAX before computing the structure factor: jax.config.update("jax_enable_x64", True)

---

# Positioning and scope

One can find extensive Blue-noise and hyperuniform sampling methods in the literature, all involving different trade-offs between spectral quality, computational cost, dimensionality, hardware requirements, implementation complexity, and support for adaptive sampling. To briefly cite some of them: **Perturbed lattices** are simple and fast; **Poisson-disk** is a mature and popular tool; **Void-and-Cluster masks** offer instant execution but are restricted to regular grids; **Relaxation methods** (such as Lloyd or CCVT) are classic but slow to converge and prone to structural artifacts; **Optimal Transport** (BNOT) provides natural adaptive sampling but is computationally heavy; **Gaussian blue noise** (GBN) achieves ultra-high spatial quality but has quadratic complexity; and **Fast reciprocal space Correlator** (FReSCo) combines ultra-high quality with near-linear complexity, though it requires installing heavy external dependencies or familiarity with docker containers.

blue-sampler is not intended as a universal replacement for these methods. It is a lightweight Python framework for experiments on blue noise and hyperuniformity, targeting simple installation (native Python code) and device flexibility (CPU/GPU), high spectral quality (achieving the commonly accepted "stealthy" criterions $S(k) \lesssim 10^{-3} - 10^{-4}$), adaptive sampling, and support for 3D and higher dimensions.


---

## 📦 Installation

```bash
pip install blue-sampler
```

---

## Quick Start

```python
import blue_sampler as blue

# Generate 10,000 2D blue-noise points
x = blue.sample_points(N=10_000, D=2)
blue.plot(x) 

# Structure factor
blue.plot_structure_factor(x) 

# Image stippling
x = blue.im2points("plots/zebra.jpg") #return points
x = blue.im2quads("plots/vangogh.jpg")   #quadrilaterals
```

---

## 🖼️ Examples 

Uniform sampling, Gaussian Blue Noise (3k points)

<p align="center">
  <img src="https://raw.githubusercontent.com/For-a-few-DPPs-more/rgbn/main/plots/huniformpoints.png" width="45%" alt="Example">
</p>

Image stippling with points (20k points)

<p align="center">
  <img src="https://raw.githubusercontent.com/For-a-few-DPPs-more/rgbn/main/plots/montage.png" width="95%" alt="Example 1">
</p>

---

## 📋 Available Samplers

### Main sampling methods

```python
x = blue.sample_points(N, D, method="rgbn") #(N, D)
```

| Method         | Description                              |
|----------------|------------------------------------------|
| `bruteforce`   | original Gaussian-Blue-Noise sampler (high quality, slow), A. G. M. Ahmed, J. Ren, and P. Wonka.  |
| `rgbn`         | Recursive Gaussian-Blue-Noise (speed-up GBN with robust approximations)      |
| `nufft`        | Non-Uniform FFT (speed-up spectral methods with fast fourier transform)    |
| `cheap`        | fast and simple lattice jittering        |

---

## Alternative Samplers

### Sobol sequence 📏

```python
x = blue.sobol(N, D) #(N, D)
```
Low-discrepancy quasi-random sequence. 

### Clusters

```python
# Raw clusters
cl = blue.sample_clusters(N, D) #(N, K, D)
blue.plot_clusters(cl) 

# Convert to point set
x = blue.cluster2points(cl) #(N, m, D)
```

### STIT Tessellations (2D only) 🧩

```python
# Raw STIT tessellation (quadrilaterals)
ts = blue.sample_tessels(N) #(N, 4, D=2)
blue.plot_tessels(ts) 

# Convert to point set
x = blue.tessel2points(ts) #(N, m, D=2)
```

### Pinwheel Tilings (2D only) 𖣘

```python
# Base pinwheel triangle
pw0 = blue.pinwheel_base()  #(3, D=2)

# Triangulation level 4
pw4 = blue.pinwheel_transform(pw0, depth=4) #(4*5**depth, 3, D=2)
blue.plot_polygons(pw4) 

#===============================
# Convert Pinwheel to point set:
#===============================
#sample points from the BASE pinwheel
x0 = blue.tessel2points(pw0) #(m, D=2)
#then replicate the sample on the full triangulation,
#in a fractal way
x4 = blue.pinwheel_transform(x0, depth = 4) #(4*5**depth, m, D=2)
```

> **Note**: The conversion from geometric objects (polygons or clusters) to point sets is performed using a standard **moment matching** technique. cluster2points(x, p = 3) and tessel2points(x, p = 3) will sample m points per batch that mimic the statistical {0, 1, ... p-1} moments of the batch.

---

## Supported Dimensions

| Dimension | Status       |
|-----------|--------------|
| 2–3D      | Fast ⚡       |
| 4–5D      | Supported    |
| ≥6D       | Experimental 🧪 |

---

## Documentation & Links 🔗

- 📖 **Documentation**: [https://blue-sampler.readthedocs.io](https://blue-sampler.readthedocs.io)
- 📦 **PyPI**: [https://pypi.org/project/blue-sampler](https://pypi.org/project/blue-sampler/)
- 🌐 **Project website**: [https://for-a-few-dpps-more.github.io/rgbn/](https://for-a-few-dpps-more.github.io/rgbn/)
- 💻 **GitHub**: [https://github.com/For-a-few-DPPs-more/rgbn](https://github.com/For-a-few-DPPs-more/rgbn)

---

## 📚 References

The algorithms and mathematical tools implemented in **blue-sampler** are based or inspired from the following works.

- **Gaussian Blue Noise (repulsive interaction kernel)**  
  *A. G. M. Ahmed, J. Ren, and P. Wonka.*  
  **Gaussian Blue Noise.**  
  *ACM Transactions on Graphics (SIGGRAPH Asia), 41(6), 2022.*  
  DOI: 10.1145/3550454.3555519

- **FReSCo (Non uniform FFT)**  
  *A. Shih, M. Casiulis, and S. Martiniani.*  
  **Fast Generation of Spectrally-Shaped Disorder.**  
  *Physical Review E*, 110(3):034122, 2024.  
  DOI: 10.1103/PhysRevE.110.034122

- **STIT tessellations and moment matching**  
  *L. Lotz and M. A. Klatt.*  
  **Persistence of asymptotic variance under transport: from hyperfluctuation to stealthy hyperuniformity.**  
  *arXiv:2605.22803*, 2026.

- **Aperiodic tiling for hyuperuniformity (here pinwheel)**  
  *A. Gabrielli, B. Jancovici, M. Joyce, J. L. Lebowitz, L. Pietronero, and F. Sylos Labini.*  
  **Generation of Primordial Cosmological Perturbations from Statistical Mechanical Models.**  
  *Physical Review D*, **67**(4):043506, 2003.  

- **SquareNet** [(v1.3.11). 2026](https://squarenet.readthedocs.io/en/latest/).  
  Grid data structure for point clouds, codeveloped with RGBN \
  for efficient neighbor query and fourier trnasform in Python.

- **Sobol sequences**  
  Wrapped from `scipy.stats.qmc.Sobol` (SciPy).

---

## License

MIT
