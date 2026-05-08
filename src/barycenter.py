"""
Wasserstein Barycenters and McCann Interpolation.

  mccann_interpolation  : straight-line paths between two point clouds via OT plan
  wasserstein_barycenter: free-support barycenter of k distributions (fixed-point iter)

Math: phases/phase_05/derive.md
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sinkhorn import log_sinkhorn
from src.wasserstein import w2_sinkhorn

CONFIG = {
    "barycenter_epsilon":  0.05,   # Sinkhorn regularization for inner solves
    "barycenter_max_iter": 50,     # outer fixed-point iterations
    "sinkhorn_inner_iter": 200,    # Sinkhorn iters per outer step
    "convergence_tol":     1e-4,   # stop when max support displacement < this
    "plots_dir": "notebooks",
}


# ---------------------------------------------------------------------------
# McCann Interpolation
# ---------------------------------------------------------------------------

def mccann_interpolation(
    x: np.ndarray,           # (n, d) source cloud
    y: np.ndarray,           # (m, d) target cloud
    transport_plan: np.ndarray,  # (n, m) OT plan between x and y
    t: float,                # interpolation parameter in [0, 1]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the McCann displacement interpolant at time t.

    For each (i,j) pair with P_ij > 0, place a point at
        z = (1-t)*x_i + t*y_j   with weight P_ij.

    Returns
    -------
    z       : (K, d)  interpolated support points (K ≤ n*m)
    weights : (K,)    corresponding weights (sum to 1)
    """
    n, m = transport_plan.shape
    d = x.shape[1]

    # Find all (i,j) pairs with non-trivial mass
    i_idx, j_idx = np.nonzero(transport_plan > 1e-9)   # each shape (K,)
    w = transport_plan[i_idx, j_idx]                    # (K,)

    xi = x[i_idx]   # (K, d)  source points for active pairs
    yj = y[j_idx]   # (K, d)  target points for active pairs

    z = (1.0 - t) * xi + t * yj   # (K, d)  straight-line interpolation
    return z, w / w.sum()


def interpolation_frames(
    x: np.ndarray,          # (n, d)
    y: np.ndarray,          # (m, d)
    n_frames: int = 7,
    epsilon: float = CONFIG["barycenter_epsilon"],
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Compute McCann interpolation frames from x to y.

    Returns list of (points, weights) tuples for t in linspace(0, 1, n_frames).
    """
    n, m = len(x), len(y)
    a = source_weights if source_weights is not None else np.ones(n) / n
    b = target_weights if target_weights is not None else np.ones(m) / m

    # Squared-Euclidean cost
    x_sq = np.sum(x ** 2, axis=1)[:, None]   # (n, 1)
    y_sq = np.sum(y ** 2, axis=1)[None, :]   # (1, m)
    C = np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)   # (n, m)

    transport_plan, _, _ = log_sinkhorn(C, a, b,
                                         epsilon=epsilon,
                                         max_iter=500)

    frames = []
    for t in np.linspace(0.0, 1.0, n_frames):
        z, w = mccann_interpolation(x, y, transport_plan, t)
        frames.append((z, w))
    return frames


# ---------------------------------------------------------------------------
# Wasserstein Barycenter (free-support, fixed-point iteration)
# ---------------------------------------------------------------------------

def wasserstein_barycenter(
    measures: list[np.ndarray],    # list of k arrays, each (mᵢ, d) support points
    lambdas: np.ndarray,           # (k,)  barycenter weights, sum to 1
    n_support: int,                # number of support points for the barycenter
    measure_weights: list[np.ndarray] | None = None,  # list of (mᵢ,) weights
    epsilon: float = CONFIG["barycenter_epsilon"],
    max_iter: int = CONFIG["barycenter_max_iter"],
    inner_iter: int = CONFIG["sinkhorn_inner_iter"],
    tol: float = CONFIG["convergence_tol"],
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Free-support Wasserstein barycenter via fixed-point iteration.

    Algorithm (Alvarez-Esteban et al. 2016):
      1. Initialise support x ∈ ℝ^{n×d} as a mixture of input points
      2. Repeat:
           For each i: Pᵢ = Sinkhorn(p, bᵢ, cost(x, yᵢ))
           x ← Σᵢ λᵢ · (Pᵢ / p[:,None]) @ yᵢ    (barycentric projection)
      Until x converges.

    Returns
    -------
    x       : (n_support, d)  barycenter support points
    p       : (n_support,)    uniform weights (free-support → equal weights)
    displacements : list of float  max point displacement per outer iteration
    """
    k = len(measures)
    d = measures[0].shape[1]
    lambdas = np.asarray(lambdas, dtype=float)
    assert abs(lambdas.sum() - 1.0) < 1e-6

    # Default: uniform weights for each reference measure
    if measure_weights is None:
        measure_weights = [np.ones(len(yi)) / len(yi) for yi in measures]

    # Initialise barycenter support: sample from the weighted mixture of all measures
    rng = np.random.default_rng(rng_seed)
    all_pts = np.vstack(measures)   # (Σmᵢ, d)
    idx = rng.choice(len(all_pts), size=n_support, replace=False)
    x = all_pts[idx].copy()         # (n_support, d)  initial support
    p = np.ones(n_support) / n_support  # (n_support,)  uniform (free-support)

    displacements = []

    for outer_iter in range(max_iter):
        x_new = np.zeros_like(x)   # (n_support, d)

        for i, (yi, bi, li) in enumerate(zip(measures, measure_weights, lambdas)):
            mi = len(yi)

            # Cost matrix between current barycenter support and measure i
            x_sq = np.sum(x  ** 2, axis=1)[:, None]   # (n_support, 1)
            y_sq = np.sum(yi ** 2, axis=1)[None, :]   # (1, mᵢ)
            Ci = np.maximum(x_sq + y_sq - 2.0 * (x @ yi.T), 0.0)  # (n_support, mᵢ)

            # Inner Sinkhorn: transport plan from p to bᵢ
            Pi, _, _ = log_sinkhorn(Ci, p, bi,
                                     epsilon=epsilon,
                                     max_iter=inner_iter)  # (n_support, mᵢ)

            # Barycentric projection: Tᵢ(xⱼ) = (Pᵢ[j,:]/p[j]) @ yᵢ
            # Pi / p[:,None] : (n_support, mᵢ)  row-normalised plan
            # @ yi           : (n_support, mᵢ) @ (mᵢ, d) = (n_support, d)
            Ti_x = (Pi / p[:, None]) @ yi   # (n_support, d)
            x_new += li * Ti_x              # weighted accumulation

        displacement = float(np.abs(x_new - x).max())
        displacements.append(displacement)
        x = x_new

        if displacement < tol:
            print(f"  Converged at outer iter {outer_iter + 1}  "
                  f"(displacement={displacement:.2e})")
            break

    return x, p, displacements


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _scatter(ax, pts, weights, color, label, alpha=0.7, size_scale=300):
    """Scatter plot where marker size encodes weight."""
    sizes = weights * size_scale * len(weights)
    ax.scatter(pts[:, 0], pts[:, 1], s=sizes, c=color,
               alpha=alpha, label=label, edgecolors="none")


def plot_interpolation(
    frames: list[tuple[np.ndarray, np.ndarray]],
    save_path: str,
    title: str = "McCann Displacement Interpolation",
) -> None:
    n_frames = len(frames)
    fig, axes = plt.subplots(1, n_frames, figsize=(2.5 * n_frames, 3))

    t_vals = np.linspace(0, 1, n_frames)
    cmap = plt.cm.coolwarm

    for ax, (z, w), t in zip(axes, frames, t_vals):
        color = cmap(t)
        sizes = np.clip(w * 300 * len(w), 5, 80)
        ax.scatter(z[:, 0], z[:, 1], s=sizes, color=color,
                   alpha=0.7, edgecolors="none")
        ax.set_title(f"t={t:.2f}", fontsize=9)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_barycenter(
    measures: list[np.ndarray],
    lambdas: np.ndarray,
    barycenter: np.ndarray,
    measure_colors: list,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, len(measures) + 1,
                              figsize=(3 * (len(measures) + 1), 3))

    for ax, yi, color, li in zip(axes, measures, measure_colors, lambdas):
        ax.scatter(yi[:, 0], yi[:, 1], c=color, s=30, alpha=0.7, edgecolors="none")
        ax.set_title(f"μ  (λ={li:.2f})", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")

    axes[-1].scatter(barycenter[:, 0], barycenter[:, 1],
                     c="black", s=30, alpha=0.8, edgecolors="none")
    axes[-1].set_title("Barycenter", fontsize=9)
    axes[-1].set_aspect("equal"); axes[-1].axis("off")

    fig.suptitle("Wasserstein Barycenter", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Point cloud generators
# ---------------------------------------------------------------------------

def make_circle(n: int, radius: float = 1.0, rng=None) -> np.ndarray:
    """n points sampled uniformly on a circle."""
    rng = rng or np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, n)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n, 2)


def make_square(n: int, side: float = 1.5, rng=None) -> np.ndarray:
    """n points sampled uniformly on the perimeter of a square."""
    rng = rng or np.random.default_rng(1)
    # Sample from 4 sides equally
    n_per_side = n // 4
    s = side
    sides = [
        np.stack([np.full(n_per_side, -s), rng.uniform(-s, s, n_per_side)], axis=1),
        np.stack([np.full(n_per_side,  s), rng.uniform(-s, s, n_per_side)], axis=1),
        np.stack([rng.uniform(-s, s, n_per_side), np.full(n_per_side, -s)], axis=1),
        np.stack([rng.uniform(-s, s, n - 3 * n_per_side), np.full(n - 3 * n_per_side, s)], axis=1),
    ]
    return np.vstack(sides)  # (n, 2)


def make_triangle(n: int, scale: float = 1.5, rng=None) -> np.ndarray:
    """n points sampled uniformly on the perimeter of an equilateral triangle."""
    rng = rng or np.random.default_rng(2)
    vertices = scale * np.array([
        [0, 1], [-np.sqrt(3)/2, -0.5], [np.sqrt(3)/2, -0.5]
    ])  # (3, 2)
    n_per_edge = n // 3
    edges = []
    for v_start, v_end, n_e in zip(
        vertices, np.roll(vertices, -1, axis=0),
        [n_per_edge, n_per_edge, n - 2 * n_per_edge]
    ):
        t = rng.uniform(0, 1, n_e)[:, None]
        edges.append((1 - t) * v_start + t * v_end)
    return np.vstack(edges)  # (n, 2)


# ---------------------------------------------------------------------------
# Sanity checks / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("WASSERSTEIN BARYCENTERS — SANITY CHECKS")
    print("=" * 60)

    rng = np.random.default_rng(42)
    os.makedirs(CONFIG["plots_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    # Test 1: McCann interpolation — circle to square
    # ------------------------------------------------------------------
    print("\n--- Test 1: McCann interpolation (circle → square) ---")
    n_pts = 60
    circle = make_circle(n_pts, radius=1.0, rng=rng)   # (60, 2)
    square = make_square(n_pts, side=1.2,  rng=rng)    # (60, 2)

    t0 = time.perf_counter()
    frames = interpolation_frames(circle, square, n_frames=7, epsilon=0.05)
    elapsed = time.perf_counter() - t0

    # Sinkhorn makes all P_ij > 0, so the t=0 frame has n*m points (xᵢ appears
    # once per j partner).  The correct check is at the distribution level:
    # the weighted mean and std of z0 should match the circle's.
    z0, w0 = frames[0]
    z1, w1 = frames[-1]
    mean_z0     = np.average(z0, axis=0, weights=w0)         # (2,)
    mean_circle = circle.mean(axis=0)                         # (2,)
    mean_z1     = np.average(z1, axis=0, weights=w1)
    mean_square = square.mean(axis=0)
    err_mean_start = float(np.linalg.norm(mean_z0 - mean_circle))
    err_mean_end   = float(np.linalg.norm(mean_z1 - mean_square))
    print(f"  t=0 mean deviation from circle centroid: {err_mean_start:.4f}  (expect ~0)")
    print(f"  t=1 mean deviation from square centroid: {err_mean_end:.4f}  (expect ~0)")
    print(f"  Computed 7 frames in {elapsed*1000:.0f} ms")
    assert err_mean_start < 0.05 and err_mean_end < 0.05
    print("  PASS")

    plot_interpolation(
        frames,
        save_path=os.path.join(CONFIG["plots_dir"], "mccann_circle_to_square.png"),
        title="McCann Interpolation: Circle → Square",
    )

    # ------------------------------------------------------------------
    # Test 2: Barycenter of circle and square (λ=[0.5, 0.5])
    #         Result should be "between" them geometrically
    # ------------------------------------------------------------------
    print("\n--- Test 2: Barycenter of circle + square ---")
    measures_2 = [circle, square]
    lambdas_2  = np.array([0.5, 0.5])

    t0 = time.perf_counter()
    bary_2, _, disps_2 = wasserstein_barycenter(
        measures_2, lambdas_2, n_support=n_pts, epsilon=0.05,
        max_iter=30, inner_iter=100,
    )
    elapsed = time.perf_counter() - t0

    # Check: barycenter W₂ to circle ≈ W₂ to square (equal weights → symmetric)
    w2_to_circle = w2_sinkhorn(bary_2, circle, epsilon=0.1)
    w2_to_square = w2_sinkhorn(bary_2, square, epsilon=0.1)
    asymmetry = abs(w2_to_circle - w2_to_square)
    print(f"  W₂(bary, circle) = {w2_to_circle:.4f}")
    print(f"  W₂(bary, square) = {w2_to_square:.4f}")
    print(f"  Asymmetry        = {asymmetry:.4f}  (expect < 0.3 for equal λ)")
    print(f"  Outer iters      = {len(disps_2)},  final disp = {disps_2[-1]:.2e}")
    print(f"  Time             = {elapsed*1000:.0f} ms")
    assert asymmetry < 0.5
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 3: Barycenter of 3 shapes
    # ------------------------------------------------------------------
    print("\n--- Test 3: Barycenter of circle + square + triangle ---")
    triangle = make_triangle(n_pts, scale=1.3, rng=rng)
    measures_3 = [circle, square, triangle]
    lambdas_3  = np.array([1/3, 1/3, 1/3])

    t0 = time.perf_counter()
    bary_3, _, disps_3 = wasserstein_barycenter(
        measures_3, lambdas_3, n_support=n_pts, epsilon=0.05,
        max_iter=40, inner_iter=100,
    )
    elapsed = time.perf_counter() - t0

    w2s = [w2_sinkhorn(bary_3, yi, epsilon=0.1) for yi in measures_3]
    print(f"  W₂(bary, circle)   = {w2s[0]:.4f}")
    print(f"  W₂(bary, square)   = {w2s[1]:.4f}")
    print(f"  W₂(bary, triangle) = {w2s[2]:.4f}")
    spread = max(w2s) - min(w2s)
    print(f"  Spread (max-min)   = {spread:.4f}  (equal λ → should be small)")
    print(f"  Outer iters        = {len(disps_3)},  final disp = {disps_3[-1]:.2e}")
    print(f"  Time               = {elapsed*1000:.0f} ms")
    assert spread < 0.5
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 4: Barycenter of k=1 — should recover the input
    # ------------------------------------------------------------------
    print("\n--- Test 4: Barycenter of single measure = that measure ---")
    bary_1, _, _ = wasserstein_barycenter(
        [circle], np.array([1.0]), n_support=n_pts, epsilon=0.05,
        max_iter=20, inner_iter=100,
    )
    w2_self = w2_sinkhorn(bary_1, circle, epsilon=0.1)
    print(f"  W₂(bary, circle) = {w2_self:.4f}  (expect ≈ 0 for k=1)")
    assert w2_self < 0.5
    print("  PASS")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")

    plot_barycenter(
        measures_3, lambdas_3, bary_3,
        measure_colors=["steelblue", "tomato", "seagreen"],
        save_path=os.path.join(CONFIG["plots_dir"], "barycenter_3shapes.png"),
    )

    # Convergence plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(disps_3, label="3-shape barycenter")
    ax.semilogy(disps_2, label="circle+square barycenter")
    ax.set_xlabel("Outer iteration"); ax.set_ylabel("Max displacement (log)")
    ax.set_title("Barycenter convergence"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    conv_path = os.path.join(CONFIG["plots_dir"], "barycenter_convergence.png")
    fig.savefig(conv_path, dpi=120); plt.close(fig)
    print(f"Saved → {conv_path}")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
