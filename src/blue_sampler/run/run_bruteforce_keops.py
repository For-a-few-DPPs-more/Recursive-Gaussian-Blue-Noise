import numpy as np
try:
    from pykeops.numpy import LazyTensor
    x = np.random.rand(10, 3).astype(np.float32)
    y = np.random.rand(10, 3).astype(np.float32)

    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    dist2 = ((x_i - y_j) ** 2).sum(-1)
    KEOPS_AVAILABLE = True
except:
    KEOPS_AVAILABLE = False


def _make_hyperparams(N, D, n_iter=60, lrbase=1.0):
    dx = N ** (-1.0 / D)
    sigma2 = 2.0 * dx**2
    high_dim = sigma2 >= 0.03

    lr = 0.1 * dx * lrbase
    scale = {2: 240, 3: 6000}.get(D, 20_000)
    n_steps = max(n_iter, int(n_iter * N / scale))

    a = 2.0 * np.pi
    b = 4.0 / (sigma2 * a**2)
    c = 1.0 / (2.0 * np.pi)

    return sigma2, lr, n_steps, high_dim, a, b, c


def _make_kernels(sigma2, high_dim, a, b, c):
    def pacman(x):
        return x - np.floor(x)

    def gaussian(x, y):
        delta = y - x
        delta -= delta.round()
        dist2 = (delta**2).sum(dim=2)
        return (delta * (-dist2 / sigma2).exp()).sum_reduction(dim=1)

    def gaussian_sine(x, y):
        delta = a * (y - x)
        sin2 = (b * (delta / 2.0).sin()**2).sum(dim=2)
        return (c * delta.sin() * (-sin2).exp()).sum_reduction(dim=1)

    name, kernel = (
        ("gaussian_sine", gaussian_sine)
        if high_dim else
        ("gaussian", gaussian)
    )
    return pacman, kernel, name


def make_pipeline(N, D, n_iter=60, lrbase=1.0, verbose=1):
    """Build a sampler for stealthy point patterns on a periodic domain.

    ``N`` points in ``D`` dimensions are optimized by gradient descent,
    with the kernel chosen automatically from the characteristic spacing.
    ``sample()`` returns the resulting ``(N, D)`` configuration in ``[0, 1)``.
    """
    assert KEOPS_AVAILABLE, "keops not working"
    sigma2, lr, n_steps, high_dim, a, b, c = _make_hyperparams(
        N, D, n_iter, lrbase
    )
    pacman, kernel, kernel_name = _make_kernels(
        sigma2, high_dim, a, b, c
    )

    def log(message, level=1):
        if verbose >= level:
            print(f"[stealthy] {message}")

    log(
        f"setup: N={N}, D={D}, σ²={sigma2:.3g}, "
        f"lr={lr:.3g}, iterations={n_steps}"
    )
    log(f"kernel: {kernel_name}")
    log("pipeline ready")

    def grad(x, idx=None):
        xi = LazyTensor(x[(slice(None) if idx is None else idx), None, :])
        xj = LazyTensor(x[None, :, :])
        return kernel(xi, xj)

    def sample(init=None):
        x = (
            np.random.rand(N, D)
            if init is None
            else np.asarray(init, dtype=np.float32).copy()
        )
        x = np.asarray(x, dtype=np.float32)

        steps = np.array([0.4, 0.7, 1.0, 1.3, 1.6])
        sample_size = min(N, 100)

        log(f"sampling: {n_steps} iterations")

        for it in range(n_steps):
            gradx = grad(x)
            gradnorm = np.linalg.norm(gradx, axis=-1).mean()

            idx = (
                np.arange(N)
                if sample_size == N
                else np.random.choice(N, sample_size, replace=False)
            )

            candidates = [
                pacman(x - step * lr * gradx / gradnorm)
                for step in steps
            ]
            scores = [
                (grad(candidate, idx) ** 2).sum()
                for candidate in candidates
            ]

            best = np.argmin(scores)
            lr *= steps[best]
            x = candidates[best]

            if verbose >= 2 and (it == 0 or (it + 1) % 10 == 0):
                log(
                    f"iteration {it + 1:>4}/{n_steps} | "
                    f"step={steps[best]:.1f} | lr={lr:.3g}"
                )

        log("Done")
        return x

    return sample