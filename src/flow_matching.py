"""
Flow Matching with OT Coupling — generative models via Optimal Transport.

Trains a velocity MLP to transport N(0,I) → target distribution
using straight-line paths defined by the mini-batch Sinkhorn coupling.

Architecture:
  VelocityMLP: (x_t, sinusoidal_emb(t)) → velocity
  Coupling:    Sinkhorn on each mini-batch cost matrix (Phase 3)
  Paths:       z_t = (1-t)x0 + t*x1  (McCann interpolation, Phase 5)
  ODE solver:  Euler integration at inference

Math: phases/phase_06/derive.md
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sinkhorn import log_sinkhorn

CONFIG = {
    # Model
    "d":           2,       # data dimensionality
    "hidden_dim":  256,     # MLP hidden layer size
    "n_hidden":    3,       # number of hidden layers
    "time_emb_dim": 16,     # sinusoidal time embedding dimension
    # Training
    "batch_size":  256,     # MLP gradient batch size
    "ot_batch":    128,     # OT coupling batch (< batch_size → cheaper Sinkhorn)
    "n_steps":     4_000,   # gradient steps; 2D task converges fast
    "lr":          1e-3,
    "lr_decay":    0.995,   # multiply lr every 100 steps
    # OT coupling
    "ot_epsilon":  0.05,    # Sinkhorn regularization for mini-batch coupling
    "ot_max_iter": 15,      # Sinkhorn iters per step — 15 is enough for approx coupling
    # Inference
    "ode_steps":   50,      # Euler steps for sample generation
    # Misc
    "seed":        42,
    "plots_dir":   "notebooks",
    "log_interval": 500,
}


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Map scalar t ∈ [0,1] to a dim-dimensional sinusoidal embedding.

    t   : (n, 1)
    out : (n, dim)

    Uses the same formula as transformer positional encoding.
    """
    half = dim // 2
    # Frequencies: 1, 1/100, 1/100², ..., (exponentially spaced)
    freqs = torch.exp(
        -math.log(100.0) * torch.arange(half, device=t.device) / (half - 1)
    )                                                     # (half,)
    args  = t * freqs[None, :]                           # (n, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (n, dim)


# ---------------------------------------------------------------------------
# Velocity network
# ---------------------------------------------------------------------------

class VelocityMLP(nn.Module):
    """
    Predicts velocity v(x, t) for the flow ODE dx/dt = v(x, t).

    Input:  x_t (d,) concatenated with sinusoidal_emb(t) (time_emb_dim,)
    Output: velocity (d,)
    """

    def __init__(
        self,
        d:           int = CONFIG["d"],
        hidden_dim:  int = CONFIG["hidden_dim"],
        n_hidden:    int = CONFIG["n_hidden"],
        time_emb_dim:int = CONFIG["time_emb_dim"],
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        in_dim = d + time_emb_dim            # x_t concat time embedding

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, d)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (n, d)    position at time t
        t : (n, 1)    time in [0, 1]
        → v : (n, d)  predicted velocity
        """
        emb = sinusoidal_embedding(t, self.time_emb_dim)   # (n, time_emb_dim)
        h   = torch.cat([x, emb], dim=-1)                  # (n, d + time_emb_dim)
        return self.net(h)                                  # (n, d)


# ---------------------------------------------------------------------------
# Mini-batch OT coupling (uses our Phase 3 Sinkhorn)
# ---------------------------------------------------------------------------

def compute_ot_coupling(
    x0: np.ndarray,   # (n, d) noise batch
    x1: np.ndarray,   # (n, d) data batch
    epsilon: float = CONFIG["ot_epsilon"],
    max_iter: int  = CONFIG["ot_max_iter"],
) -> np.ndarray:
    """
    Compute the Sinkhorn transport plan between two equal-size batches.

    Cost: squared Euclidean distance C_ij = ‖x0_i − x1_j‖²
    Returns P : (n, n)  transport plan with uniform marginals 1/n
    """
    n = len(x0)
    a = np.ones(n) / n    # (n,) uniform source
    b = np.ones(n) / n    # (n,) uniform target

    # Squared-Euclidean cost matrix
    x0_sq = np.sum(x0 ** 2, axis=1)[:, None]   # (n, 1)
    x1_sq = np.sum(x1 ** 2, axis=1)[None, :]   # (1, n)
    C = np.maximum(x0_sq + x1_sq - 2.0 * (x0 @ x1.T), 0.0)  # (n, n)

    P, _, _ = log_sinkhorn(C, a, b, epsilon=epsilon, max_iter=max_iter)
    return P   # (n, n)


def sample_ot_pairs(
    P: np.ndarray,    # (n, n) transport plan
    x0: np.ndarray,   # (n, d) noise batch
    x1: np.ndarray,   # (n, d) data batch
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each noise point x0_i, sample one paired target x1_j ~ P[i,:].

    Vectorized: use numpy's cumulative-sum + uniform trick instead of a
    Python for-loop over rows (which is O(n) Python calls and very slow).
    """
    n = len(x0)
    # Normalise rows → probability distributions
    row_probs = P / P.sum(axis=1, keepdims=True)    # (n, n)

    # Vectorized multinomial: for each row, draw one index
    # cumsum along columns; compare to uniform → gives the sampled index
    cdf = np.cumsum(row_probs, axis=1)              # (n, n)
    u   = rng.uniform(size=(n, 1))                  # (n, 1)
    # First column where cdf > u gives the sampled j
    j_indices = (cdf < u).sum(axis=1)              # (n,)  vectorised
    j_indices = np.clip(j_indices, 0, n - 1)
    return x0, x1[j_indices]


# ---------------------------------------------------------------------------
# Data generators (2D toy distributions)
# ---------------------------------------------------------------------------

def make_dataset(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate 2D toy datasets.

    name: 'moons' | '8gaussians' | 'circles' | 'checkerboard'
    Returns (n, 2) array of samples, zero-centred and unit-ish scale.
    """
    if name == "moons":
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=n, noise=0.05, random_state=int(rng.integers(1000)))
        X = (X - X.mean(0)) / X.std()

    elif name == "8gaussians":
        centres = np.array([
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1/np.sqrt(2), 1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)),
            (1/np.sqrt(2), -1/np.sqrt(2)), (-1/np.sqrt(2), -1/np.sqrt(2))
        ]) * 2.0   # (8, 2)
        idx = rng.integers(0, 8, size=n)
        X = centres[idx] + rng.normal(0, 0.3, size=(n, 2))

    elif name == "circles":
        from sklearn.datasets import make_circles
        X, _ = make_circles(n_samples=n, noise=0.03, factor=0.5,
                             random_state=int(rng.integers(1000)))
        X = (X - X.mean(0)) / X.std()

    elif name == "checkerboard":
        x = rng.uniform(-2, 2, size=n * 4)
        y = rng.uniform(-2, 2, size=n * 4)
        mask = ((np.floor(x) + np.floor(y)) % 2 == 0)
        pts  = np.stack([x[mask], y[mask]], axis=1)
        idx  = rng.choice(len(pts), size=n, replace=False)
        X    = pts[idx]

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return X.astype(np.float32)   # (n, 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_name: str   = "moons",
    use_ot:       bool  = True,
    n_steps:      int   = CONFIG["n_steps"],
    batch_size:   int   = CONFIG["batch_size"],
    device:       str   = "cpu",
    seed:         int   = CONFIG["seed"],
) -> tuple["VelocityMLP", list[float]]:
    """
    Train a flow matching velocity network.

    use_ot=True  : OT-coupled FM (Sinkhorn mini-batch coupling)
    use_ot=False : independent coupling (random pairing)

    Returns trained model and loss history.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = VelocityMLP().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=100, gamma=CONFIG["lr_decay"]
    )

    # Pre-generate a large pool of data points; sample mini-batches from it
    n_pool = 50_000
    data_pool = make_dataset(dataset_name, n_pool, rng)   # (50000, 2)

    losses = []
    t_start = time.perf_counter()

    for step in range(1, n_steps + 1):
        model.train()

        # --- Sample mini-batch from data pool (full training batch)
        idx_full = rng.choice(n_pool, size=batch_size, replace=False)
        x1_full  = data_pool[idx_full]                                  # (batch_size, 2)
        x0_full  = rng.normal(0, 1, size=(batch_size, 2)).astype(np.float32)

        # --- OT coupling on a smaller sub-batch (ot_batch << batch_size)
        #     then tile the paired indices up to the full batch_size.
        ot_n = CONFIG["ot_batch"]   # e.g. 128
        if use_ot:
            # Use first ot_n samples for coupling; repeat to fill batch_size
            x0_ot = x0_full[:ot_n]
            x1_ot = x1_full[:ot_n]
            P = compute_ot_coupling(x0_ot, x1_ot,
                                     epsilon=CONFIG["ot_epsilon"],
                                     max_iter=CONFIG["ot_max_iter"])
            x0_paired, x1_paired = sample_ot_pairs(P, x0_ot, x1_ot, rng)
            # Tile to reach batch_size
            repeats = math.ceil(batch_size / ot_n)
            x0_np = np.tile(x0_paired, (repeats, 1))[:batch_size]
            x1_np = np.tile(x1_paired, (repeats, 1))[:batch_size]
        else:
            x0_np, x1_np = x0_full, x1_full
        # --- To torch
        x0 = torch.from_numpy(x0_np).to(device)   # (n, 2)
        x1 = torch.from_numpy(x1_np).to(device)   # (n, 2)

        # --- Sample t ~ U[0, 1]
        t = torch.rand(batch_size, 1, device=device)   # (n, 1)

        # --- Interpolate: z_t = (1-t)*x0 + t*x1  (McCann interpolation)
        z_t = (1.0 - t) * x0 + t * x1                 # (n, 2)

        # --- Constant target velocity along the straight path
        v_target = x1 - x0                            # (n, 2)

        # --- Predict and loss
        v_pred = model(z_t, t)                         # (n, 2)
        loss   = F.mse_loss(v_pred, v_target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        losses.append(loss.item())

        if step % CONFIG["log_interval"] == 0:
            avg = np.mean(losses[-CONFIG["log_interval"]:])
            elapsed = time.perf_counter() - t_start
            label = "OT" if use_ot else "indep"
            print(f"  [{label}] step {step:5d} | loss {avg:.4f} | "
                  f"{elapsed:.1f}s elapsed")

    return model, losses


# ---------------------------------------------------------------------------
# Sampling (Euler ODE integration)
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(
    model:    VelocityMLP,
    n:        int,
    device:   str = "cpu",
    n_steps:  int = CONFIG["ode_steps"],
) -> np.ndarray:
    """
    Generate n samples by integrating dx/dt = v_θ(x,t) from t=0 to t=1.

    x(0) ~ N(0, I)  →  x(1) ≈ samples from target distribution

    Returns (n, d) numpy array.
    """
    model.eval()
    x = torch.randn(n, CONFIG["d"], device=device)   # (n, 2)  start from noise
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i * dt
        t     = torch.full((n, 1), t_val, device=device)   # (n, 1)
        v     = model(x, t)                                  # (n, 2) velocity
        x     = x + dt * v                                   # Euler step

    return x.cpu().numpy()   # (n, 2)


@torch.no_grad()
def get_paths(
    model:  VelocityMLP,
    x0:     torch.Tensor,   # (k, 2) starting points
    device: str = "cpu",
    n_steps:int = 20,
) -> np.ndarray:
    """
    Trace the ODE path for k starting points.
    Returns array of shape (n_steps+1, k, 2).
    """
    model.eval()
    x = x0.to(device)
    dt = 1.0 / n_steps
    trajectory = [x.cpu().numpy()]

    for i in range(n_steps):
        t = torch.full((len(x), 1), i * dt, device=device)
        x = x + dt * model(x, t)
        trajectory.append(x.cpu().numpy())

    return np.stack(trajectory, axis=0)   # (n_steps+1, k, 2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(
    data:          np.ndarray,   # (n, 2) real data
    samples_ot:    np.ndarray,   # (n, 2) OT-coupled FM samples
    samples_indep: np.ndarray,   # (n, 2) independent FM samples
    save_path:     str,
    dataset_name:  str,
) -> None:
    """Side-by-side: real data, OT-coupled generated, independent generated."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles  = ["Real data", "OT-coupled FM", "Independent FM"]
    samples_list = [data, samples_ot, samples_indep]
    colors  = ["steelblue", "tomato", "seagreen"]

    lim = max(np.abs(data).max(), 3.5) * 1.1
    for ax, pts, title, c in zip(axes, samples_list, titles, colors):
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.4, c=c)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Flow Matching — {dataset_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_paths(
    paths_ot:    np.ndarray,   # (T, k, 2)
    paths_indep: np.ndarray,   # (T, k, 2)
    save_path:   str,
) -> None:
    """Show ODE paths for a handful of starting points for both couplings."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    titles = ["OT-coupled paths", "Independent paths"]

    for ax, paths, title in zip(axes, [paths_ot, paths_indep], titles):
        k = paths.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for i in range(k):
            traj = paths[:, i, :]              # (T, 2)
            ax.plot(traj[:, 0], traj[:, 1],
                    color=colors[i], alpha=0.7, linewidth=1.5)
            ax.scatter(traj[0, 0],  traj[0, 1],  marker="o",
                       color=colors[i], s=40, zorder=3)
            ax.scatter(traj[-1, 0], traj[-1, 1], marker="*",
                       color=colors[i], s=80, zorder=3)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    axes[0].set_title("OT-coupled: straighter paths")
    axes[1].set_title("Independent: paths may cross")
    fig.suptitle("ODE trajectories (circle=start, star=end)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_losses(
    losses_ot:    list[float],
    losses_indep: list[float],
    save_path:    str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    window = 100
    def smooth(x): return np.convolve(x, np.ones(window)/window, mode="valid")
    ax.plot(smooth(losses_ot),    label="OT-coupled",   color="tomato")
    ax.plot(smooth(losses_indep), label="Independent",  color="steelblue")
    ax.set_xlabel("Training step")
    ax.set_ylabel("FM loss (smoothed)")
    ax.set_title("Training loss: OT-coupled vs independent")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("FLOW MATCHING WITH OT COUPLING")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    rng = np.random.default_rng(CONFIG["seed"])

    os.makedirs(CONFIG["plots_dir"], exist_ok=True)

    for dataset_name in ["moons", "8gaussians"]:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        # ------------------------------------------------------------------
        # Train OT-coupled and independent FM side by side
        # ------------------------------------------------------------------
        print("\nTraining OT-coupled flow matching...")
        model_ot, losses_ot = train(
            dataset_name=dataset_name,
            use_ot=True,
            n_steps=CONFIG["n_steps"],
            device=device,
            seed=CONFIG["seed"],
        )

        print("\nTraining independent (random-coupled) flow matching...")
        model_indep, losses_indep = train(
            dataset_name=dataset_name,
            use_ot=False,
            n_steps=CONFIG["n_steps"],
            device=device,
            seed=CONFIG["seed"],
        )

        # ------------------------------------------------------------------
        # Generate samples
        # ------------------------------------------------------------------
        n_gen = 2000
        print(f"\nGenerating {n_gen} samples from each model...")
        samples_ot    = euler_sample(model_ot,    n_gen, device)
        samples_indep = euler_sample(model_indep, n_gen, device)
        real_data     = make_dataset(dataset_name, n_gen, rng)

        # ------------------------------------------------------------------
        # ODE path comparison (20 starting points, 20 steps)
        # ------------------------------------------------------------------
        x0_fixed = torch.randn(20, 2)   # same starting points for both models
        paths_ot    = get_paths(model_ot,    x0_fixed, device, n_steps=20)
        paths_indep = get_paths(model_indep, x0_fixed, device, n_steps=20)

        # Measure path "straightness": how curved is the trajectory?
        # Straight path: endpoint - startpoint is constant direction.
        # Curvature proxy: sum of squared velocity changes (jerk).
        def path_straightness(paths):
            # paths: (T, k, 2)
            # velocity at each step: paths[t+1] - paths[t]
            vel = np.diff(paths, axis=0)              # (T-1, k, 2)
            jerk = np.diff(vel, axis=0)               # (T-2, k, 2) velocity change
            return float(np.mean(jerk ** 2))          # mean squared jerk

        straight_ot    = path_straightness(paths_ot)
        straight_indep = path_straightness(paths_indep)
        print(f"\nPath straightness (lower = straighter):")
        print(f"  OT-coupled:  {straight_ot:.4e}")
        print(f"  Independent: {straight_indep:.4e}")
        print(f"  OT is {straight_indep/straight_ot:.1f}× straighter than independent")

        # ------------------------------------------------------------------
        # Plots
        # ------------------------------------------------------------------
        plot_comparison(
            real_data, samples_ot, samples_indep,
            save_path=os.path.join(CONFIG["plots_dir"],
                                   f"fm_samples_{dataset_name}.png"),
            dataset_name=dataset_name,
        )
        plot_paths(
            paths_ot, paths_indep,
            save_path=os.path.join(CONFIG["plots_dir"],
                                   f"fm_paths_{dataset_name}.png"),
        )
        plot_losses(
            losses_ot, losses_indep,
            save_path=os.path.join(CONFIG["plots_dir"],
                                   f"fm_loss_{dataset_name}.png"),
        )

    print("\n" + "=" * 60)
    print("Phase 6 complete.")
    print("=" * 60)
