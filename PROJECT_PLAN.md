# PROJECT_PLAN.md — Optimal Transport from Scratch

## Goal

Build a complete, from-scratch understanding of Optimal Transport: the mathematics,
the algorithms, and a real application. Every algorithm is derived before it is coded.
Every phase ends with a course chapter written while the derivations are fresh.

---

## Rules (non-negotiable)

- `scipy.optimize`, `numpy`, `torch` only until the algorithm exists in this codebase
- `pot` is used only for cross-validation after a scratch implementation is complete
- Every algorithm: worked numerical example with real numbers traced through
- Every matrix op: shape trace in comment
- Order in every phase: derive → implement → validate → course chapter

---

## Phase Breakdown

### Phase 0 — Setup
**Deliverables:** conda env, directory skeleton, git init, GitHub repo, `.gitignore`
**New concepts:** none

---

### Phase 1 — Discrete OT: The Linear Program
**Deliverables:**
- `phases/phase_01/derive.md` — Monge problem → Kantorovich relaxation → LP formulation
- `src/lp_ot.py` — exact LP solver via `scipy.optimize.linprog`
- A 3×3 example solved by hand, then verified with the solver
- Cross-validate against `pot.emd()` (after solver exists)
- `course/01_discrete_ot.md`

**New concepts:**
- Monge's map and why it can fail to exist
- Kantorovich's relaxation: joint distributions instead of maps
- The cost matrix C, transport plan P, marginal constraints
- LP formulation: min ⟨C, P⟩ s.t. P1 = a, Pᵀ1 = b, P ≥ 0
- Why this is always feasible (product measure a⊗b is a feasible point)

---

### Phase 2 — Kantorovich Duality
**Deliverables:**
- `phases/phase_02/derive.md` — dual LP derivation, c-transform, complementary slackness
- `src/dual_ot.py` — solve the dual, verify primal == dual value at optimum
- Geometric visualization of dual variables as "prices"
- `course/02_duality.md`

**New concepts:**
- LP duality: from the primal transport plan to dual potentials (u, v)
- Dual problem: max ⟨a,u⟩ + ⟨b,v⟩ s.t. uᵢ + vⱼ ≤ Cᵢⱼ
- Strong duality (proved via LP theory, no measure theory needed)
- The c-transform: v = u^c means vⱼ = minᵢ(Cᵢⱼ − uᵢ)
- Complementary slackness: Pᵢⱼ > 0 ⟹ uᵢ + vⱼ = Cᵢⱼ
- Economic interpretation: dual variables as shipping prices

---

### Phase 3 — The Sinkhorn Algorithm (Entropic OT)
**Deliverables:**
- `phases/phase_03/derive.md` — entropy regularization, derive Sinkhorn fixed point, log-domain form
- `src/sinkhorn.py` — vanilla Sinkhorn and log-domain Sinkhorn
- Convergence plot vs ε, stability comparison
- Cross-validate against `pot.sinkhorn()` (after scratch impl exists)
- `course/03_sinkhorn.md`

**New concepts:**
- Entropy-regularized OT: min ⟨C,P⟩ + ε·KL(P ‖ a⊗b) s.t. marginals
- Why entropy helps: strict convexity, unique solution, GPU-friendly
- Deriving the Sinkhorn iterations from KKT conditions: P = diag(u)·K·diag(v), K = exp(−C/ε)
- The fixed-point iteration: u ← a/(Kv), v ← b/(Kᵀu)
- Log-domain formulation for numerical stability
- ε tradeoff: small ε → exact OT but unstable; large ε → smooth but blurred
- Convergence rate: linear in ε (Hilbert metric contraction)

---

### Phase 4 — Wasserstein Distances
**Deliverables:**
- `phases/phase_04/derive.md` — W_p definition, 1D closed form, metric properties
- `src/wasserstein.py` — W₁ and W₂ (1D closed form via quantile, 2D via Sinkhorn)
- `src/sliced_wasserstein.py` — Monte Carlo sliced approximation
- Comparison: exact vs sliced on 2D Gaussians
- `course/04_wasserstein.md`

**New concepts:**
- W_p(μ,ν) = (min_P ∫ d(x,y)^p dP)^(1/p) — the Wasserstein-p distance
- Metric properties: positivity, symmetry, triangle inequality
- 1D closed form: W_p(μ,ν) = ‖F_μ⁻¹ − F_ν⁻¹‖_p (quantile functions)
- Why W_2 on Gaussians has a closed form (Bures metric)
- Sliced Wasserstein: average W_p over random 1D projections
- Why Wasserstein is better than KL/JS for learning (geometry of the space)

---

### Phase 5 — Wasserstein Barycenters & Interpolation
**Deliverables:**
- `phases/phase_05/derive.md` — Fréchet mean in W₂ space, McCann interpolation, Sinkhorn barycenter
- `src/barycenter.py` — iterative Sinkhorn barycenter (Cuturi & Doucet algorithm)
- Visual demo: interpolating between 2D point clouds
- `course/05_barycenters.md`

**New concepts:**
- The Fréchet mean in Wasserstein space: argmin_μ Σᵢ λᵢ W₂²(μ, μᵢ)
- McCann interpolation: the displacement interpolant between two measures
- Sinkhorn barycenter: alternating projection algorithm
- Fixed-point iteration for barycenters
- Convergence and uniqueness (under W₂)

---

### Phase 6 — Flow Matching: Generative Models via Optimal Transport
**Deliverables:**
- `phases/phase_06/derive.md` — the flow matching objective, OT coupling, why straight paths are optimal under W₂
- `src/flow_matching.py` — OT-coupled flow matching: Sinkhorn coupling + MLP velocity network + Euler ODE sampler
- Experiments on 2D toy distributions: Gaussian → moons, Gaussian → 8-Gaussians
- Trajectory visualization: show how paths straighten when OT coupling is used vs random coupling
- Optional MNIST experiment if it fits in 4GB VRAM
- `course/06_flow_matching.md`

**New concepts:**
- The continuous normalizing flow (CNF) framing: learn a vector field v(x, t) s.t. dX/dt = v(X, t)
- Flow matching objective: regress v onto straight-line velocities (x_data − x_noise)
- Why straight paths are W₂-optimal: the OT map between Gaussians IS a straight-line flow
- OT coupling: use Sinkhorn (Phase 3) to pair each noise sample with its nearest data sample, reducing path crossings
- Rectified flow (Liu et al. 2022) and flow matching (Lipman et al. 2022) as two views of the same idea
- Why this replaced diffusion: fewer NFE (neural function evaluations), straighter paths, faster sampling

**Connection to your diffusion model background:**
Diffusion models learn to reverse a fixed noisy SDE. Flow matching learns to follow an OT-optimal ODE directly.
The DDPM objective is a special case of flow matching with a specific (non-optimal) coupling and noise schedule.
Using OT coupling makes the paths straighter → faster inference → why SD3 / Flux switched to it.

---

## Component Table: Scratch vs Library

| Component | Implementation | Library Used After |
|---|---|---|
| Discrete OT (LP) | `src/lp_ot.py` (scipy.optimize.linprog) | `pot.emd()` — cross-val only |
| Dual OT | `src/dual_ot.py` (scipy.optimize.linprog) | — |
| Sinkhorn (vanilla) | `src/sinkhorn.py` (numpy) | `pot.sinkhorn()` — cross-val only |
| Sinkhorn (log-domain) | `src/sinkhorn.py` (numpy) | — |
| W₁, W₂ (1D) | `src/wasserstein.py` (numpy) | — |
| Sliced Wasserstein | `src/sliced_wasserstein.py` (numpy) | — |
| Wasserstein Barycenter | `src/barycenter.py` (numpy) | — |
| Flow Matching (velocity net) | `src/flow_matching.py` (torch) | — |

---

## Directory Structure

```
ot-learn/
├── CLAUDE.md
├── PROJECT_PLAN.md
├── src/
│   ├── lp_ot.py
│   ├── dual_ot.py
│   ├── sinkhorn.py
│   ├── wasserstein.py
│   ├── sliced_wasserstein.py
│   └── barycenter.py
├── phases/
│   ├── phase_00/
│   ├── phase_01/
│   │   ├── derive.md
│   │   └── *.py
│   ├── phase_02/ ...
│   └── phase_06/
├── notebooks/          # PNG plots (matplotlib.use("Agg"))
├── checkpoints/        # Not tracked
├── data/               # Not tracked
└── course/
    ├── 01_discrete_ot.md
    ├── 02_duality.md
    ├── 03_sinkhorn.md
    ├── 04_wasserstein.md
    ├── 05_barycenters.md
    └── 06_application.md
```

---

## Hardware Notes

All phases through Phase 5 run on CPU — no GPU needed. The Sinkhorn barycenter
demo (Phase 5) may be slow for large point clouds; will use small synthetic data.

Phase 6 hardware notes:
- **Option A (Color Transfer)**: CPU-only. Images resize to ~50k pixels — fine.
- **Option B (WGAN-GP)**: GPU required. MNIST/Fashion-MNIST fits easily in 4GB.
- **Option C (Word Mover's Distance)**: CPU-only. Pre-trained embeddings (~800MB RAM).
- **Option D (Single-Cell)**: CPU-only. Small public datasets (~10k cells).

---

## Math Depth

Algorithmic rigor: full derivations of all algorithms and duality, LP proofs,
Sinkhorn convergence (Hilbert metric), Wasserstein metric properties.
No measure theory (no Radon measures, no Brenier's theorem proof).
Brenier's theorem will be stated and given geometric intuition in Phase 5.

---

## Course Timing

A course chapter is written at the end of every phase, before Phase N+1 begins.
