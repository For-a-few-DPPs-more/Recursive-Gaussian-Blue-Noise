# Tests

```bash
pip install -e ".[test]"
pytest                          # fast suite (~1 min on CPU)
pytest -m slow                  # quality of the samples (structure factor at low k)
pytest -m "slow or not slow"    # everything
```

| File                       | What                                                              |
|----------------------------|-------------------------------------------------------------------|
| `test_sample_points.py`    | every method in D = 2, 3, 4 (N = 128), warm starts, argument checks |
| `test_targets.py`          | point-set and image targets, frames, `image_to_points`            |
| `test_warmstart.py`        | `warmstart_points`, the lattices, Sobol                           |
| `test_extras.py`           | clusters, tessellations, pinwheel, `tile`, extras plots           |
| `test_structure_factor.py` | structure factor, plots; `slow`: blue noise vs Poisson, stealthiness |
| `test_internals.py`        | torus balls, moment matching, config, progress, `check_gpu`       |

Known issue, marked `xfail`: `cstit` can return points slightly outside [0, 1)
(moment matching is not constrained to the cube). Once fixed, these tests show up as XPASS: remove the marker in `test_sample_points.py`.
