# Hyperuniform datasets

This subfolder contains point samples with high-quality spectra. The folder itself
contains only the `small datasets`; larger datasets can be downloaded separately from
GitHub Large File Storage (TODO).

## RGBN datasets

All datasets marked `rgbn` were generated using the
`blue_sampler` package with:

```python
blue_sampler.samplepoints(method="rgbn")
```

The datasets were generated on a 10 GB GPU (Google Colab T4), with a typical
runtime of less than 10 minutes per dataset.

## FINUFFT datasets

All datasets marked `finufft` were generated using the
sampling procedure provided in the finuft.ipynb notebook.

The datasets were generated on a 10 GB GPU (Google Colab T4), with a typical
runtime of less than 10 minutes per dataset.
