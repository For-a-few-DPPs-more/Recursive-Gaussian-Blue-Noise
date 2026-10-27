import numpy as np

try:
    from pykeops.numpy import LazyTensor

    _x = np.random.rand(10, 3).astype(np.float32)
    _y = np.random.rand(10, 3).astype(np.float32)
    _dist2 = ((LazyTensor(_x[:, None, :]) - LazyTensor(_y[None, :, :])) ** 2).sum(-1)
    KEOPS_AVAILABLE = True
    KEOPS_ERROR = None
except Exception as e:
    KEOPS_AVAILABLE = False
    KEOPS_ERROR = e


LEARNING_RATE = {2: 0.3, 3: 0.1, 4: 0.05, 5: 0.01}


def _default_steps(N, D):
    if D == 2:
        return max(20, N // 4)
    if D == 3:
        return max(20, N // 100)
    return 20


def _make_kernels(sigma):
    sigma2 = sigma**2

    def min_image(xi, xj):
        delta = xi - xj
        delta -= delta.round()
        dist2 = (delta**2).sum(dim=2)
        return (delta * (-dist2 / (2.0 * sigma2)).exp()).sum_reduction(dim=1)

    def sinusoidal(xi, xj):
        delta = xi - xj
        s2 = ((np.pi * delta).sin() ** 2).sum(dim=2) / (2.0 * np.pi**2 * sigma2)
        w = (-s2).exp()
        return ((2.0 * np.pi * delta).sin() * w / (2.0 * np.pi)).sum_reduction(dim=1)

    if sigma < 0.1:
        return min_image, "min_image"
    return sinusoidal, "sinusoidal"


def make_pipeline(N, D,  n_steps=None, lr=1.0, verbose=1):
    assert KEOPS_AVAILABLE, f"keops not working: {KEOPS_ERROR}"

    s = 1.0
    
    delta = N ** (-1.0 / D)
    sigma = s * delta
    eta = LEARNING_RATE.get(D, 0.001) * delta**2
    coef = eta / sigma**2
    T = _default_steps(N, D) if n_steps is None else n_steps
    kernel, kernel_name = _make_kernels(sigma)

    def log(message, level=1):
        if verbose >= level:
            print(f"[gbn] {message}")

    log(f"setup: N={N}, D={D}, sigma={sigma:.3g}, eta={eta:.3g}, T={T}")
    log(f"kernel: {kernel_name}")

    def force(x):
        return kernel(LazyTensor(x[:, None, :]), LazyTensor(x[None, :, :]))

    def sample(init=None):
        x = np.array(
            np.random.rand(N, D) if init is None else init,
            dtype=np.float32,
            order="C",
        )
        for t in range(T):
            x = (x + lr * coef * force(x)) % 1.0
            x = np.ascontiguousarray(x, dtype=np.float32)
            if verbose >= 2 and (t == 0 or (t + 1) % 10 == 0):
                log(f"iteration {t + 1:>4}/{T}")
        return x

    return sample