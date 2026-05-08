"""
Wasserstein distances.

  w1_1d / w2_1d       : exact 1D W₁, W₂ via quantile (O(n log n))
  w2_gaussian         : W₂ between Gaussians, closed form (Bures metric)
  w2_sinkhorn         : W₂ approximation via Sinkhorn (any dimension)
  sliced_wasserstein  : sliced W_p via random projections (high-dimensional)

Math: phases/phase_04/derive.md
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sinkhorn import log_sinkhorn

CONFIG = {
    "sinkhorn_epsilon": 0.1,       # regularization for w2_sinkhorn
    "sinkhorn_max_iter": 500,      # fewer iters: ε=0.1 converges in ~100 steps
    "sliced_n_projections": 200,   # number of random directions for sliced W
    "sliced_rng_seed": 0,
    "plots_dir": "notebooks",
}


# ---------------------------------------------------------------------------
# 1D Wasserstein via quantile function
# ---------------------------------------------------------------------------

def w1_1d(
    x: np.ndarray,  # (n,) source samples (or positions)
    y: np.ndarray,  # (m,) target samples (or positions)
    a: np.ndarray | None = None,  # (n,) source weights; None → uniform
    b: np.ndarray | None = None,  # (m,) target weights; None → uniform
) -> float:
    """
    Exact W₁ in 1D via the quantile formula.

    For equal-weight case: sort both, pair by rank, average |xᵢ − yᵢ|.
    For unequal weights:  build the empirical quantile functions and integrate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)

    if a is None:
        a = np.ones(n) / n   # (n,) uniform
    if b is None:
        b = np.ones(m) / m   # (m,)

    # Build empirical CDFs: sort and compute cumulative weights
    x_order = np.argsort(x)
    y_order = np.argsort(y)
    x_sorted, a_sorted = x[x_order], a[x_order]   # (n,) each
    y_sorted, b_sorted = y[y_order], b[y_order]   # (m,)

    # Merge both CDFs onto a common grid of quantile levels
    # CDF breakpoints: cumulative weight just before each mass
    a_cdf = np.concatenate([[0.0], np.cumsum(a_sorted)])   # (n+1,)
    b_cdf = np.concatenate([[0.0], np.cumsum(b_sorted)])   # (m+1,)

    # All CDF breakpoints merged
    all_levels = np.unique(np.concatenate([a_cdf, b_cdf]))  # sorted

    # Evaluate F_μ⁻¹(t) and F_ν⁻¹(t) at each level via searchsorted
    # np.searchsorted(a_cdf, t, side='right') - 1 gives the index of the
    # atom just below quantile t
    def quantile_val(sorted_pts, cdf, t):
        idx = np.searchsorted(cdf, t, side="right") - 1
        idx = np.clip(idx, 0, len(sorted_pts) - 1)
        return sorted_pts[idx]

    w1 = 0.0
    for i in range(len(all_levels) - 1):
        t_mid = 0.5 * (all_levels[i] + all_levels[i + 1])
        dt = all_levels[i + 1] - all_levels[i]
        qx = quantile_val(x_sorted, a_cdf, t_mid)
        qy = quantile_val(y_sorted, b_cdf, t_mid)
        w1 += abs(qx - qy) * dt   # ∫|F_μ⁻¹ − F_ν⁻¹| dt

    return float(w1)


def w2_1d(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> float:
    """Exact W₂ in 1D: sqrt(∫|F_μ⁻¹(t) − F_ν⁻¹(t)|² dt)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)

    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m

    x_order, y_order = np.argsort(x), np.argsort(y)
    x_sorted, a_sorted = x[x_order], a[x_order]
    y_sorted, b_sorted = y[y_order], b[y_order]

    a_cdf = np.concatenate([[0.0], np.cumsum(a_sorted)])
    b_cdf = np.concatenate([[0.0], np.cumsum(b_sorted)])
    all_levels = np.unique(np.concatenate([a_cdf, b_cdf]))

    def quantile_val(sorted_pts, cdf, t):
        idx = np.clip(np.searchsorted(cdf, t, side="right") - 1, 0, len(sorted_pts) - 1)
        return sorted_pts[idx]

    w2_sq = 0.0
    for i in range(len(all_levels) - 1):
        t_mid = 0.5 * (all_levels[i] + all_levels[i + 1])
        dt = all_levels[i + 1] - all_levels[i]
        qx = quantile_val(x_sorted, a_cdf, t_mid)
        qy = quantile_val(y_sorted, b_cdf, t_mid)
        w2_sq += (qx - qy) ** 2 * dt   # ∫|F_μ⁻¹ − F_ν⁻¹|² dt

    return float(np.sqrt(w2_sq))


# ---------------------------------------------------------------------------
# W₂ between Gaussians (closed form)
# ---------------------------------------------------------------------------

def w2_gaussian(
    mean1: np.ndarray,  # (d,)
    cov1:  np.ndarray,  # (d,d)
    mean2: np.ndarray,  # (d,)
    cov2:  np.ndarray,  # (d,d)
) -> float:
    """
    W₂ between N(mean1, cov1) and N(mean2, cov2) via the Bures metric.

    W₂² = ‖mean1 − mean2‖²  +  B(cov1, cov2)²
    B²  = Tr(cov1) + Tr(cov2) − 2·Tr((cov1^{1/2} cov2 cov1^{1/2})^{1/2})

    For 1D (scalar covariances σ₁², σ₂²):
      W₂² = (μ₁−μ₂)² + (σ₁−σ₂)²
    """
    mean1 = np.asarray(mean1, dtype=float)
    mean2 = np.asarray(mean2, dtype=float)
    cov1  = np.asarray(cov1,  dtype=float)
    cov2  = np.asarray(cov2,  dtype=float)

    mean_term = float(np.sum((mean1 - mean2) ** 2))   # ‖m₁-m₂‖²

    # Matrix square root of cov1 via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov1)            # cov1 = V Λ Vᵀ
    eigvals = np.maximum(eigvals, 0.0)                 # numerical safety
    sqrt_cov1 = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T   # (d,d)

    # Compute (sqrt_cov1 @ cov2 @ sqrt_cov1)^{1/2}
    M = sqrt_cov1 @ cov2 @ sqrt_cov1                  # (d,d)
    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    eigvals_M = np.maximum(eigvals_M, 0.0)
    sqrt_M = eigvecs_M @ np.diag(np.sqrt(eigvals_M)) @ eigvecs_M.T  # (d,d)

    bures_sq = (np.trace(cov1) + np.trace(cov2)
                - 2.0 * np.trace(sqrt_M))              # scalar
    w2_sq = mean_term + float(bures_sq)
    return float(np.sqrt(max(w2_sq, 0.0)))             # clip for numerics


# ---------------------------------------------------------------------------
# W₂ via Sinkhorn (any dimension)
# ---------------------------------------------------------------------------

def w2_sinkhorn(
    x: np.ndarray,   # (n, d) source samples
    y: np.ndarray,   # (m, d) target samples
    epsilon: float = CONFIG["sinkhorn_epsilon"],
    max_iter: int   = CONFIG["sinkhorn_max_iter"],
    a: np.ndarray | None = None,  # (n,) weights
    b: np.ndarray | None = None,  # (m,)
) -> float:
    """
    Approximate W₂ using Sinkhorn with squared-Euclidean cost.

    Cost matrix: C_ij = ‖xᵢ − yⱼ‖²    shape (n, m)

    Returns sqrt of the Sinkhorn transport cost (the W₂ approximation).
    """
    x = np.asarray(x, dtype=float)  # (n, d)
    y = np.asarray(y, dtype=float)  # (m, d)
    n, m = len(x), len(y)

    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m

    # Squared Euclidean cost matrix
    # ‖xᵢ−yⱼ‖² = ‖xᵢ‖² − 2 xᵢ·yⱼ + ‖yⱼ‖²
    x_sq = np.sum(x ** 2, axis=1)[:, None]  # (n, 1)
    y_sq = np.sum(y ** 2, axis=1)[None, :]  # (1, m)
    C = x_sq + y_sq - 2.0 * (x @ y.T)      # (n, m)
    C = np.maximum(C, 0.0)                  # clip floating-point negatives

    _, sinkhorn_cost, _ = log_sinkhorn(C, a, b,
                                        epsilon=epsilon,
                                        max_iter=max_iter)
    return float(np.sqrt(max(sinkhorn_cost, 0.0)))


# ---------------------------------------------------------------------------
# Sliced Wasserstein distance
# ---------------------------------------------------------------------------

def sliced_wasserstein(
    x: np.ndarray,   # (n, d) source samples
    y: np.ndarray,   # (m, d) target samples
    p: int = 2,
    n_projections: int = CONFIG["sliced_n_projections"],
    rng_seed: int = CONFIG["sliced_rng_seed"],
) -> float:
    """
    Sliced Wasserstein distance SW_p(μ, ν).

    For each of n_projections random unit vectors θ ∈ Sᵈ⁻¹:
      1. Project: x_proj = x @ θ  (n,),  y_proj = y @ θ  (m,)
      2. Compute 1D W_p via quantile formula
    Return (mean of W_p^p)^{1/p}.

    x: (n, d)
    y: (m, d)
    """
    x = np.asarray(x, dtype=float)   # (n, d)
    y = np.asarray(y, dtype=float)   # (m, d)
    d = x.shape[1]
    assert y.shape[1] == d

    rng = np.random.default_rng(rng_seed)
    # Sample random unit vectors: draw from Gaussian, normalise
    thetas = rng.standard_normal((n_projections, d))   # (L, d)
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True)  # (L, d) unit vectors

    _w1d = w1_1d if p == 1 else w2_1d
    wp_values = np.empty(n_projections)

    for l in range(n_projections):
        x_proj = x @ thetas[l]   # (n,)  1D projections
        y_proj = y @ thetas[l]   # (m,)
        wp_values[l] = _w1d(x_proj, y_proj) ** p

    return float(np.mean(wp_values) ** (1.0 / p))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_1d_transport(
    x: np.ndarray, y: np.ndarray,
    title: str, save_path: str,
) -> None:
    """Visualise 1D OT: show sorted pairings between source and target."""
    xs = np.sort(x)
    ys = np.sort(y)
    n = min(len(xs), len(ys))

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.scatter(xs, np.zeros_like(xs), s=60, color="steelblue", zorder=3, label="source")
    ax.scatter(ys, np.ones_like(ys),  s=60, color="tomato",    zorder=3, label="target")
    for i in range(n):
        ax.plot([xs[i], ys[i]], [0, 1], "gray", alpha=0.5, linewidth=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["source", "target"])
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_sliced_vs_exact(
    w_exact_list: list[float],
    w_sliced_list: list[float],
    n_list: list[int],
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_list, w_exact_list,  "o-", label="Exact W₂ (Sinkhorn)")
    ax.plot(n_list, w_sliced_list, "s--", label="Sliced W₂")
    ax.set_xlabel("Number of samples n")
    ax.set_ylabel("W₂ estimate")
    ax.set_title("Sliced vs exact W₂ (2D Gaussians)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Sanity checks / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("WASSERSTEIN DISTANCES — SANITY CHECKS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: W₁ 1D — uniform shift of 1, hand-traced
    # ------------------------------------------------------------------
    print("\n--- Test 1: W₁ 1D, hand-traced example from derive.md ---")
    x_ex = np.array([1.0, 3.0, 5.0, 7.0])
    y_ex = np.array([2.0, 4.0, 6.0, 8.0])
    w1_val = w1_1d(x_ex, y_ex)
    print(f"  W₁({list(x_ex)}, {list(y_ex)}) = {w1_val:.6f}  (expected 1.0)")
    assert abs(w1_val - 1.0) < 1e-6, f"got {w1_val}"
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 2: W₂ 1D — same shift
    # ------------------------------------------------------------------
    print("\n--- Test 2: W₂ 1D, same shift-by-1 example ---")
    w2_val = w2_1d(x_ex, y_ex)
    print(f"  W₂ = {w2_val:.6f}  (expected 1.0 — equal spacing so same as W₁)")
    assert abs(w2_val - 1.0) < 1e-6
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 3: W₂ Gaussian closed form — 1D case
    # ------------------------------------------------------------------
    print("\n--- Test 3: W₂ Gaussian closed form (1D) ---")
    # N(0,1) vs N(3,4) → W₂² = (0-3)² + (1-2)² = 9 + 1 = 10 → W₂ = √10
    w2_gauss = w2_gaussian(
        mean1=np.array([0.0]), cov1=np.array([[1.0]]),
        mean2=np.array([3.0]), cov2=np.array([[4.0]]),
    )
    expected = np.sqrt(10.0)
    print(f"  W₂(N(0,1), N(3,4)) = {w2_gauss:.6f}  (expected {expected:.6f})")
    assert abs(w2_gauss - expected) < 1e-5
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 4: W₂ Sinkhorn ≈ Gaussian closed form
    # ------------------------------------------------------------------
    print("\n--- Test 4: Sinkhorn W₂ approximates Gaussian closed form ---")
    rng = np.random.default_rng(42)
    # n=200: cost matrix is 200×200 → fast at ε=0.1
    n_samples = 200
    x_gauss = rng.normal(0.0, 1.0, size=(n_samples, 1))   # (200, 1) N(0,1)
    y_gauss = rng.normal(3.0, 2.0, size=(n_samples, 1))   # (200, 1) N(3,4)

    t0 = time.perf_counter()
    w2_sink = w2_sinkhorn(x_gauss, y_gauss, epsilon=0.1)
    elapsed = time.perf_counter() - t0

    print(f"  Sinkhorn W₂ = {w2_sink:.4f}")
    print(f"  Exact W₂    = {expected:.4f}")
    print(f"  Error       = {abs(w2_sink - expected):.4f}  (expect < 0.3 for n=200)")
    print(f"  Solve time  = {elapsed*1000:.1f} ms")
    assert abs(w2_sink - expected) < 0.3
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 5: Sliced W₂ on 2D Gaussians
    # ------------------------------------------------------------------
    print("\n--- Test 5: Sliced W₂ on 2D Gaussians ---")
    # N([0,0], I) vs N([2,1], I).  True W₂ = ‖[2,1]‖ = √5 ≈ 2.236.
    #
    # Sliced W₂ projects onto random unit vectors θ ∈ S¹ and averages.
    # For unit-covariance Gaussians shifted by μ, the 1D projection gives
    # W₂(proj) = |θᵀμ|.  E[|θᵀμ|²] = ‖μ‖²/d, so SW₂ ≈ W₂/√d.
    # In 2D: SW₂ ≈ 2.236/√2 ≈ 1.58.  This is CORRECT — sliced systematically
    # underestimates by 1/√d.  We test the expected range, not exact equality.
    n2 = 500
    x_2d = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=n2)  # (500,2)
    y_2d = rng.multivariate_normal([2, 1], [[1, 0], [0, 1]], size=n2)  # (500,2)
    true_w2_2d = np.sqrt(5.0)                    # ‖[2,1]‖ = 2.236
    expected_sw2 = true_w2_2d / np.sqrt(2.0)     # ≈ 1.581 for d=2

    t0 = time.perf_counter()
    sw2 = sliced_wasserstein(x_2d, y_2d, p=2, n_projections=200)
    elapsed_sw = time.perf_counter() - t0

    t0 = time.perf_counter()
    exact_w2_2d = w2_sinkhorn(x_2d, y_2d, epsilon=0.1)
    elapsed_exact = time.perf_counter() - t0

    print(f"  True W₂                   = {true_w2_2d:.4f}")
    print(f"  Expected sliced W₂ (≈W₂/√d) = {expected_sw2:.4f}")
    print(f"  Exact Sinkhorn W₂         = {exact_w2_2d:.4f}  ({elapsed_exact*1000:.0f} ms)")
    print(f"  Sliced W₂                 = {sw2:.4f}  ({elapsed_sw*1000:.0f} ms)")
    print(f"  Sliced vs expected error  = {abs(sw2 - expected_sw2):.4f}")
    # Sliced underestimates by ~1/√d in d dims; allow 0.3 tolerance around that
    assert abs(sw2 - expected_sw2) < 0.3, f"sliced={sw2:.4f} expected≈{expected_sw2:.4f}"
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 6: symmetry — W(μ,ν) == W(ν,μ)
    # ------------------------------------------------------------------
    print("\n--- Test 6: symmetry W₁(μ,ν) == W₁(ν,μ) ---")
    x_sym = rng.uniform(0, 5, size=50)
    y_sym = rng.uniform(2, 8, size=50)
    w_fwd = w1_1d(x_sym, y_sym)
    w_rev = w1_1d(y_sym, x_sym)
    print(f"  W₁(x→y) = {w_fwd:.6f},  W₁(y→x) = {w_rev:.6f}")
    assert abs(w_fwd - w_rev) < 1e-10
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 7: degenerate case — identical distributions → W = 0
    # ------------------------------------------------------------------
    print("\n--- Test 7: W(μ, μ) = 0 ---")
    z = rng.normal(0, 1, size=100)
    w_self = w1_1d(z, z)
    print(f"  W₁(z, z) = {w_self:.2e}  (expected 0)")
    assert w_self < 1e-10
    print("  PASS")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")
    os.makedirs(CONFIG["plots_dir"], exist_ok=True)

    plot_1d_transport(
        x_ex, y_ex,
        title="1D OT: source {1,3,5,7} → target {2,4,6,8}  (W₁=1.0)",
        save_path=os.path.join(CONFIG["plots_dir"], "wasserstein_1d_transport.png"),
    )

    # Sliced vs exact over increasing n
    n_vals = [30, 60, 100, 150]
    w_exact_vals, w_sliced_vals = [], []
    for n_v in n_vals:
        xv = rng.multivariate_normal([0, 0], np.eye(2), size=n_v)
        yv = rng.multivariate_normal([2, 1], np.eye(2), size=n_v)
        w_exact_vals.append(w2_sinkhorn(xv, yv, epsilon=0.05))
        w_sliced_vals.append(sliced_wasserstein(xv, yv, p=2))

    plot_sliced_vs_exact(
        w_exact_vals, w_sliced_vals, n_vals,
        save_path=os.path.join(CONFIG["plots_dir"], "wasserstein_sliced_vs_exact.png"),
    )

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
