# Hyperuniform datasets

This subfolder contains point samples with high-quality spectra. The folder itself
contains only small datasets; larger datasets can be downloaded separately from
GitHub Large File Storage (TODO).

## RGBN datasets

All datasets marked `rgbn` were **generated for this repository** using the
`blue_sampler` package with:

```python
blue_sampler.samplepoints(method="rgbn")
```

The datasets were generated on a 10 GB GPU (Google Colab T4), with a typical
runtime of less than 10 minutes per dataset.

## NUFFT datasets

All datasets marked `nufft` were **generated for this repository** using the
sampling procedure provided in the following notebook:

[NUFFT sampling notebook (Google Colab)](https://colab.research.google.com/drive/1QDc6u6qYDsEXze5uoFEDNGIfgNfAEJip?usp=sharing&utm_source=chatgpt.com)

The datasets were generated on a 10 GB GPU (Google Colab T4), with a typical
runtime of less than 10 minutes per dataset.


## 200k_2D_FIRE_princeton dataset

This dataset is **not original to this repository**. It was originally published by
Morse et al. as supplementary data for their paper on large disordered stealthy
hyperuniform systems.

The dataset is redistributed here for convenience. The original dataset is also
available from the [Princeton Data Commons](https://doi.org/10.34770/e49n-r807) with *double precision* (instead of numpy float64 precision)
under the MIT License.

**Original source:**

> Morse, Peter K., Kim, Jaeuk, Steinhardt, Paul, & Torquato, Salvatore. (2023).
> *Data for "Generating large disordered stealthy hyperuniform systems with
> ultra-high accuracy to determine their physical properties".*
> [Data set]. Version 1. Princeton University.
> https://doi.org/10.34770/e49n-r807

Please cite the original dataset and its associated publication when using
`200k_2D_FIRE_princeton`.
