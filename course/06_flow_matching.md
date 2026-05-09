# Chapter 6 — Flow Matching: Generative Models via Optimal Transport

## The Problem and Why It Matters

We've spent five phases building Optimal Transport from scratch.
Here's why it all matters.

**Flow matching** is the generative modelling framework used in Stable Diffusion 3,
Flux, and Sora. Its core mechanism is this: train a neural network to move samples
from a simple noise distribution N(0,I) to a complex data distribution (images, audio,
molecular structures) by following straight-line paths.

The OT coupling from Phase 3 (Sinkhorn) defines which noise sample gets paired with
which data sample, turning those straight paths into W₂-geodesics (Phase 5 McCann
interpolation). Everything we've built is present.

**Comparison to diffusion models:**
Diffusion models (e.g. DDPM, Stable Diffusion 1/2) learn to generate data by reversing a noisy blurring process — a stochastic random walk from data to noise, then learned in reverse. They work well but are slow: generating one sample requires 50–1000 forward passes through a neural network.

- Diffusion: noisy stochastic reverse process, 50-1000 sampling steps
- Flow matching: deterministic ODE, 10-50 steps, straighter paths, same quality

---

## The Setup

Goal: learn a vector field `v_θ(x, t)` such that integrating the ODE:

```
dx/dt = v_θ(x, t),    x(0) ~ N(0, I)
```

from t=0 to t=1 transforms noise samples into samples from a target distribution p₁.

No score function, no noise schedule, no variance-preserving SDE — just a vector
field and an ODE.

---

## The Flow Matching Objective

The exact marginal vector field that transports p₀ → p₁ is intractable to compute.
The key insight (Lipman et al., 2022): define a **conditional** flow for each
source-target pair (x₀, x₁) and regress onto that.

**Conditional path** between one noise point x₀ and one data point x₁:

```
z_t = (1 − t)·x₀ + t·x₁       straight line, t ∈ [0, 1]
```

The velocity along this path is **constant**:

```
dz_t/dt = x₁ − x₀
```

**The flow matching loss:**

```
L(θ) = E_{t ~ U[0,1]}  E_{(x₀, x₁) ~ q}  ‖ v_θ(z_t, t) − (x₁ − x₀) ‖²
```

This asks: predict the straight-line velocity from wherever you are at time t.

**Why this works:** it can be shown that minimising this conditional objective gives
the same gradient as minimising the true marginal flow objective — even though the
marginal objective is intractable. The conditional and marginal objectives are
equivalent in expectation.

---

## The Coupling q(x₀, x₁) Matters

The coupling q decides which noise point pairs with which data point. It doesn't
change *what* distribution we learn to generate — both the OT and independent
couplings converge to p₁ — but it dramatically changes how **easy** the velocity
field is to learn.

```
Text diagram:

Independent coupling (random pairing):
  noise → ●──────────╲──────► data
  noise →  ●────────────╲────► data
  noise →   ●───────╲──────────► data
             paths cross → complex velocity field

OT coupling (Sinkhorn pairing):
  noise → ●────────────────────► data
  noise →  ●───────────────────► data
  noise →   ●──────────────────► data
             no crossings → simpler velocity field
```

**Quantitatively (from our experiment):**

| Dataset   | OT path jerk | Indep path jerk | OT speedup |
|-----------|-------------|----------------|------------|
| Moons     | 6.8e-07     | 4.1e-05        | **60.8×**  |
| 8-Gaussians | 4.1e-07   | 5.7e-05        | **139.6×** |

OT-coupled paths are 61–140× straighter — they barely curve at all.

**Why straighter paths help:**
1. Lower variance gradient signal → faster training
2. Fewer Euler steps needed at inference (paths barely curve)
3. The velocity field has less to "decide" — it points in nearly the same direction
   everywhere along a given path

**Training loss:**
```
OT-coupled FM:   loss 0.048  (moons),   0.072  (8-gaussians)
Independent FM:  loss 1.333  (moons),   2.079  (8-gaussians)
```

The OT loss is ~28× lower on moons, ~29× lower on 8-gaussians. The network is
fitting a fundamentally simpler function.

---

## Mini-Batch OT Coupling

We can't compute the OT plan over the full dataset at each step (too expensive).
Instead, for each mini-batch:

```
1. Sample x₀ ~ N(0,I)     (ot_batch=128 points)
2. Sample x₁ ~ data       (ot_batch=128 points)
3. Build C_ij = ‖x₀_i − x₁_j‖²    shape (128, 128)
4. P = Sinkhorn(uniform, uniform, C, ε=0.05)    shape (128, 128)
5. For each i: sample j ~ P[i,:] / P[i,:].sum()
6. Paired batch: (x₀_i, x₁_{j(i)})    shape (128, 2) each
```

Step 5 vectorised (no Python for-loop!):

```python
cdf = np.cumsum(row_probs, axis=1)    # (n, n)  running sum along targets
u   = rng.uniform(size=(n, 1))        # (n, 1)  one uniform per source
j   = (cdf < u).sum(axis=1)           # (n,)   index of first CDF ≥ u
```

This gives unbiased estimates of the true OT coupling as batch size → ∞.
With batch_size=128 and 15 Sinkhorn iterations: ~35ms per step on CPU.

---

## Architecture: The Velocity MLP

Input: `[x_t (2D)] + [sinusoidal_emb(t) (16D)]` → concatenate → `(18,)`
Hidden: 3 × Linear(256) + SiLU activation
Output: Linear → `(2,)` velocity

**Sinusoidal time embedding:**

```python
def sinusoidal_embedding(t, dim=16):
    half = dim // 2
    freqs = exp(-log(100) * arange(half) / (half-1))    # (8,)
    args  = t * freqs[None, :]                           # (n, 8)
    return cat([sin(args), cos(args)], dim=-1)           # (n, 16)
```

This gives t a rich representation: the network sees both low-frequency (global
time position) and high-frequency (fine-grained time) components.
Same idea as positional encoding in your transformer project.

**SiLU activation:** `x · σ(x)` — smooth, non-monotone, better than ReLU for
fitting smooth velocity fields that are continuous in t.

---

## Training Algorithm

```python
for step in 1..n_steps:
    # 1. Sample data and noise
    x1 = sample_from_dataset(batch_size)        # (n, 2)
    x0 = randn(batch_size, 2)                   # (n, 2)

    # 2. OT coupling: pair x0[i] with its nearest x1[j]
    C = pairwise_sq_dist(x0[:128], x1[:128])    # (128, 128)
    P = sinkhorn(C, eps=0.05, iters=15)         # (128, 128)
    x0, x1 = sample_pairs_from_plan(P, x0, x1) # (n, 2) each

    # 3. Sample interpolation time
    t = rand(n, 1)                              # (n, 1)  ~ U[0,1]

    # 4. Straight-line interpolation (McCann, Phase 5)
    z_t = (1-t)*x0 + t*x1                      # (n, 2)

    # 5. Target velocity — constant along path
    v_target = x1 - x0                         # (n, 2)

    # 6. Predict and backprop
    v_pred = v_theta(z_t, t)                   # (n, 2)
    loss   = mse(v_pred, v_target)
    loss.backward(); optimiser.step()
```

---

## Inference: Euler ODE Integration

```python
x = randn(n, 2)                   # start from noise
dt = 1 / n_steps

for i in range(n_steps):
    t  = full(n, i * dt)
    v  = v_theta(x, t)            # predicted velocity at (x, t)
    x += dt * v                   # Euler step

return x                          # ≈ samples from p₁
```

With OT coupling, paths barely curve → 50 Euler steps is more than enough.
Without OT, you might need 200+ to get the same quality (paths curve more,
the Euler discretisation accumulates more error).

---

## Results

### Moons dataset

```
OT-coupled:  loss=0.048  path jerk=6.8e-07   [see notebooks/fm_samples_moons.png]
Independent: loss=1.333  path jerk=4.1e-05   OT is 61× straighter
```

Both models generate recognisable moons. The OT model's paths are nearly straight
lines from noise to data. The independent model's paths curve significantly.

### 8-Gaussians dataset

```
OT-coupled:  loss=0.072  path jerk=4.1e-07   [see notebooks/fm_samples_8gaussians.png]
Independent: loss=2.079  path jerk=5.7e-05   OT is 140× straighter
```

On the multi-modal 8-gaussians, the advantage is larger: independent coupling
creates massive path crossings across the 8 modes, while OT assigns each noise
point to its nearest mode.

---

## The Six-Phase Connection

Every phase contributed to this result:

```
Phase 1 (LP)         → conceptual foundation: what does "optimal" mean?
Phase 2 (Duality)    → ∇u(x₀) IS the initial velocity; dual potentials = flow
Phase 3 (Sinkhorn)   → computes the mini-batch coupling at every training step
Phase 4 (Wasserstein)→ OT coupling minimises W₂²(noise_batch, data_batch)
Phase 5 (McCann)     → z_t = (1-t)x₀ + tx₁ IS the displacement interpolant
Phase 6 (This phase) → regression onto those interpolant velocities = FM
```

---

## Shape Traces: Full Forward Pass

```
x_t     : (n, d)          = (256, 2)
t       : (n, 1)          = (256, 1)

# Sinusoidal embedding
freqs   : (time_emb_dim/2,)= (8,)
args    : (n, 8)          = t * freqs
emb     : (n, 16)         = [sin(args), cos(args)]

# MLP
h₀      : (n, 18)         = cat([x_t, emb])
h₁      : (n, 256)        = SiLU(W₁ h₀ + b₁)
h₂      : (n, 256)        = SiLU(W₂ h₁ + b₂)
h₃      : (n, 256)        = SiLU(W₃ h₂ + b₃)
v_pred  : (n, 2)          = W₄ h₃ + b₄
```

---

## Summary Table

| Concept | Definition |
|---|---|
| Flow matching | Regress v_θ onto straight-line conditional velocities |
| Conditional path | z_t = (1−t)x₀ + tx₁ |
| Conditional velocity | u = x₁ − x₀ (constant) |
| OT coupling | Sinkhorn plan pairs x₀ with nearest x₁ (Phase 3) |
| Mini-batch OT | Sinkhorn on each batch; unbiased, 128×128, 15 iters |
| Path straightness | Measured by "jerk" (squared velocity change) |
| Euler integration | x ← x + dt·v_θ(x,t); 50 steps at inference |
| Independent coupling | Random pairing; 61-140× more curved paths |
