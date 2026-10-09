import time
from math import pi
from typing import NamedTuple

import numpy as np
from ..gpu_setup import set_config  # builds the compute "config" (CPU/GPU, float32/64, NUFFT backend)


# =============================================================================
# Wave-vector selection (D = 1, 2, 3 only)
# =============================================================================
# On optimise les positions via un ensemble fini de modes de Fourier k ∈ ℤ^D.
# Comme fk(-k) = conj(fk(k)), on ne garde qu'un représentant par paire ±k
# (demi-espace). On en prend exactement M = round(Chi * D * (N-1)),
# les M plus petits |k|² (égalité de |k|² → tirage aléatoire).
# Chaque mode est pondéré par 1/|k|² dans la loss (basses fréquences prioritaires).


class WaveVectors(NamedTuple):
    """Conteneur des modes de Fourier sélectionnés."""
    k: np.ndarray        # (M, D) int32 – un représentant par paire ±k
    r2: np.ndarray       # (M,)   int64 – |k|²
    w: np.ndarray        # (M,)   float64 – poids de loss ∈ (0, 1], max = 1
    M: int               # nombre de modes indépendants
    chi_eff: float       # M / (D*(N-1)) ≈ Chi


def _ball_volume_coeff(D):
    """Coefficient de volume de la boule unité en dimension D (2, π ou 4π/3)."""
    return {1: 2.0, 2: pi, 3: 4.0 * pi / 3.0}[D]


def _half_space_mask(k):
    """
    Masque booléen : True si la première coordonnée non-nulle de k est > 0.
    Garantit exactement un représentant par paire ±k (l'origine est exclue).
    """
    pos = np.zeros(len(k), dtype=bool)
    decided = np.zeros(len(k), dtype=bool)
    for d in range(k.shape[1]):
        nz = ~decided & (k[:, d] != 0)
        pos |= nz & (k[:, d] > 0)
        decided |= nz
    return pos


def get_wave_vectors(N, D, Chi, rng=None):
    """
    Sélectionne les M modes de Fourier de plus basse fréquence.

    Objectif
    --------
    On veut contraindre les M ≈ Chi·D·(N-1) modes les plus bas.
    On construit donc une "boule" de rayons croissants dans ℤ^D jusqu'à
    avoir assez de points, on ne garde que le demi-espace (un k par ±k),
    puis on trie par |k|² et on coupe à M.

    Étapes
    ------
    1. Estimation du rayon R ≈ (2M / vol(boule unité))^{1/D}.
    2. Énumération du cube entier [-R,R]^D → on ne retient que les
       points du demi-espace situés dans la boule |k|² ≤ R².
    3. Si on a trop peu de modes, on agrandit R et on recommence.
    4. Tri par |k|² (égalité → ordre aléatoire) et conservation des M premiers.
    5. Poids w = 1/|k|² (normalisés max=1) pour privilégier les basses fréquences.
    """
    rng = np.random.default_rng() if rng is None else rng

    n_eff = N - 1                       # k=0 exclu → N-1 degrés de liberté
    M = int(round(Chi * D * n_eff))
    if M < 1:
        raise ValueError(f"Chi={Chi} gives no mode for N={N}, D={D}")

    # Estimation initiale du rayon de la boule qui contient ~M paires ±k
    R = int(np.ceil((2.0 * M / _ball_volume_coeff(D)) ** (1.0 / D))) + 1

    while True:
        # Cube entier [-R … R]^D
        axis = np.arange(-R, R + 1, dtype=np.int32)
        grids = np.meshgrid(*([axis] * D), indexing="ij")
        k = np.stack([g.ravel() for g in grids], axis=1)
        r2 = np.sum(k.astype(np.int64) ** 2, axis=1)

        # On garde uniquement le demi-espace ET l'intérieur de la boule
        keep = (r2 <= R * R) & _half_space_mask(k)
        if keep.sum() >= M:
            break
        R += max(1, R // 10)            # pas assez de modes → on agrandit

    k, r2 = k[keep], r2[keep]

    # Tri : d'abord |k|², puis aléatoire en cas d'égalité (dernière coquille)
    order = np.lexsort((rng.random(len(r2)), r2))[:M]
    k, r2 = k[order], r2[order]

    # Poids inversement proportionnels à |k|² (basses fréquences prioritaires)
    w = 1.0 / (r2.astype(np.float64) + 1e-3)
    w /= w.max()

    chi_eff = M / (D * n_eff)
    return WaveVectors(k=k, r2=r2, w=w, M=M, chi_eff=chi_eff)


# =============================================================================
# NUFFT optimisation
# =============================================================================

def _nufft_pipeline(N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
                    n_iter=120, precision="float32", device="auto", seed=None,
                    verbose=1):
    """
    Optimisation NUFFT pour générer un nuage de points à faible discrépance dans [0,1)^D.

    Principe
    --------
    On minimise l'énergie de Fourier
        L(x) = (1/M) Σ_k  w_k |f(k) − f_target(k)|²
    où f(k) = Σ_j exp(i 2π k·x_j) est le facteur de structure.
    Les gradients sont évalués en O(N log N) via NUFFT (type-1 et type-2).

    Paramètres
    ----------
    N : int
        Nombre de points.
    D : int
        Dimension (1, 2 ou 3 uniquement).
    lr : float
        Échelle du pas initial ≈ lr · 0.1 · N^(−1/D).
    warmstart : array (N, D) ou None
        Configuration de départ (sinon tirage uniforme).
    Chi : float
        Fraction de degrés de liberté contraints (modes retenus).
        ≤ 0.3 → basses fréquences seulement (rapide)
        ≈ 0.4 → équilibre (défaut)
        ≥ 0.5 → cristallisation possible
    target : array ou str, optionnel
        Densité cible (nuage de points ou chemin d'image). None → uniforme.
    n_iter : int
        Nombre max d'itérations de descente de gradient.
    precision : {"float32", "float64"}
        float64 plus lent mais permet d'atteindre ~10⁻²⁰.
    device : {"cpu", "cuda", "auto"}
    seed : int ou None
    verbose : int
        >0 → affiche la progression.

    Retour
    ------
    x : ndarray (N, D)
        Positions optimisées dans [0,1)^D.
    """
    if D not in (1, 2, 3):
        raise ValueError(f"Only D=1,2,3 supported (got {D})")

    cfg = set_config(device, precision, verbose)

    xp = cfg.xp
    real_dtype = cfg.real_dtype
    complex_dtype = cfg.complex_dtype
    device = cfg.device
    precision = cfg.precision
    to_numpy = cfg.to_numpy
    nufft_lib = cfg.nufft_lib

    eps = 1e-4 if precision == "float32" else 1e-8
    rng = np.random.default_rng(seed)

    # Positions initiales
    if warmstart is not None:
        x = xp.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        x = xp.asarray(rng.uniform(size=(N, D)), dtype=real_dtype)

    # Sélection des modes
    wv = get_wave_vectors(N, D, Chi, rng)
    M, chi_eff = wv.M, wv.chi_eff

    # Grille FFT minimale (paire) qui contient tous les modes sélectionnés
    K_max = int(np.abs(wv.k).max())
    G = 2 * (K_max + 1)
    n_modes = (G,) * D

    # Fréquences en ordre FFT : 0, 1, …, G/2-1, −G/2, …, −1
    freqs_np = np.concatenate([np.arange(0, G // 2), np.arange(-(G // 2), 0)])
    freqs = xp.asarray(freqs_np, dtype=real_dtype)
    if D == 1:
        ks = (freqs,)
    else:
        ks = xp.meshgrid(*[freqs] * D, indexing="ij")

    # Masque de poids : non-nul uniquement sur les M modes retenus
    w_np = np.zeros(n_modes, dtype=np.float64)
    w_np[tuple(wv.k[:, d].astype(np.int64) % G for d in range(D))] = wv.w
    w = xp.asarray(w_np, dtype=real_dtype)

    norm = float(M)

    # Plans NUFFT réutilisables (type-1 : points → modes, type-2 : modes → points)
    plan_kwargs = dict(
        n_trans=1,
        eps=eps,
        isign=1,
        dtype=complex_dtype,
        modeord=1,
    )
    plan1 = nufft_lib.Plan(1, n_modes, **plan_kwargs)
    plan2 = nufft_lib.Plan(2, n_modes, **plan_kwargs)

    c = xp.ones(N, dtype=complex_dtype)  # poids unitaires

    # Spectre cible (calculé une seule fois)
    if target is not None:
        if isinstance(target, str):          # chemin d'image
            target = im2spectrum(target, shape=n_modes, invert=True)
            fk_target = xp.asarray(target, dtype=complex_dtype)
            scale = 1.0 / N
        else:                                # nuage de points
            target = xp.asarray(target, dtype=real_dtype)
            if target.ndim != 2 or target.shape[1] != D:
                raise ValueError(f"target must have shape (M, {D}), got {target.shape}")
            M_t = target.shape[0]
            c_target = xp.ones(M_t, dtype=complex_dtype)
            coords_t = tuple(2.0 * xp.pi * target[:, d] for d in range(D))
            plan1.setpts(*coords_t)
            fk_target = plan1.execute(c_target) / M_t
            scale = 1.0 / N
    else:
        fk_target = 0.0
        scale = 1.0

    def loss_and_grad(x):
        """Calcule la loss et le gradient ∇_x L pour les positions courantes."""
        # NUFFT attend des coordonnées dans [−π, π)
        coords = tuple(2.0 * xp.pi * x[:, d] for d in range(D))

        # Passage avant (type-1) : f(k)
        plan1.setpts(*coords)
        fk = plan1.execute(c) * scale

        diff = fk - fk_target
        loss = float(xp.sum(xp.abs(diff) ** 2 * w) / norm)

        # Passage arrière (type-2) : gradient par composante
        grads = []
        for d in range(D):
            grad_source = (2.0 * w * xp.conj(diff) * (1j * 2.0 * xp.pi * ks[d])).astype(
                complex_dtype
            )
            plan2.setpts(*coords)
            g_d = plan2.execute(grad_source)
            grads.append(g_d.real * scale)

        grad = xp.stack(grads, axis=1) / norm
        return loss, grad

    # Pas initial ≈ fraction de la distance moyenne inter-points
    delta = lr * 0.1 * N ** (-1.0 / D)

    if verbose:
        tgt_info = f"target={target.shape[0]} pts" if target is not None else "uniform"
        print(
            f"[nufft | {device}] "
            f"N={N}  D={D}   Chi={Chi:.3f} (eff={chi_eff:.5f})  "
            f"modes(+-k pairs)={M}  G={G}  "
            f"n_iter={n_iter}   ({tgt_info})"
        )
        print("For Early-stopping : Ctrl-C (Keyboard interrupt ⏹️)")

    t0 = time.time()
    adaptive = delta
    prev_loss = float("inf")

    try:
        for it in range(n_iter):
            loss, grad = loss_and_grad(x)

            # Normalisation RMS → le pas contrôle uniquement la longueur du déplacement
            rms = xp.sqrt(xp.mean(grad**2) + 1e-30)
            x_new = x - (grad / rms) * adaptive
            x_new = x_new - xp.floor(x_new)      # repli périodique dans [0,1)^D

            # Adaptation du pas ("bold driver")
            if loss < prev_loss:
                x = x_new
                adaptive *= 1.05
                prev_loss = loss
            else:
                adaptive *= 0.8
                x = x_new

            if verbose and (it % 20 == 0 or it == n_iter - 1):
                print(f"  iter {it:4d} | loss {loss:.1e}")

    except KeyboardInterrupt:
        if verbose:
            print(f"\n  [KeyboardInterrupt] stopped at iter {it} | loss {loss:.4f}")
            print(f"  elapsed : {time.time() - t0:.1f}s")

    if verbose:
        print(f"  done in {time.time() - t0:.1f}s")

    return to_numpy(x)


def im2spectrum(path, shape=(512, 512), invert=True):
    """
    Charge une image et renvoie son spectre de Fourier (DFT d'une densité normalisée).

    - Conversion en niveaux de gris, redimensionnement à `shape`.
    - Orientation : origine en bas à gauche (convention mathématique).
    - Si invert=True, les pixels sombres correspondent à une densité élevée.
    - Normalisation → densité de probabilité (somme = 1).

    Retour
    ------
    spectrum : ndarray complexe de shape `shape`
    """
    from PIL import Image

    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float64) / 255.0
    rho = (rho.T)[::-1]                 # origine en bas à gauche
    if invert:
        rho = 1.0 - rho                 # sombre → densité forte
    rho = rho / rho.sum()
    spectrum = np.fft.fftn(rho)
    return spectrum
