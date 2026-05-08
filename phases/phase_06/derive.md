# Phase 6 Derivation — Flow Matching via Optimal Transport

## The Big Picture

We've spent five phases building OT from scratch. Here's the payoff.

Flow matching is the generative modelling framework behind Stable Diffusion 3,
Flux, and Sora. Its core mechanism is exactly the McCann displacement interpolation
from Phase 5, applied between random noise and real data. The OT coupling from
Phase 3 (Sinkhorn) makes the paths straight, which makes the velocity field simpler
to learn.

Every tool we've built is present:
- Phase 1 (LP): conceptual grounding of OT
- Phase 2 (Duality): the dual potentials ARE the initial velocity
- Phase 3 (Sinkhorn): computes the mini-batch OT coupling during training
- Phase 4 (Wasserstein): the optimality criterion for straight paths
- Phase 5 (McCann): the interpolation IS the training path

---

## 1. The Problem: Learning to Transport Noise to Data

Goal: given samples from a complex data distribution p₁ (e.g. 2D moons),
learn to transform samples from a simple source p₀ = N(0,I) into p₁.

**Diffusion models** do this via a noisy SDE, learning to reverse a stochastic
blurring process. Works but requires 50-1000 denoising steps at inference.

**Flow matching** does this via a deterministic ODE:
    dx/dt = v_θ(x, t),    x(0) ~ p₀

If we train v_θ well, running this ODE from t=0 to t=1 transforms samples from
p₀ into samples from p₁. No noise, no Markov chains — just a vector field.

---

## 2. The Flow Matching Objective

We want v_θ to "match" the true velocity field that transports p₀ to p₁.

The exact marginal vector field is intractable to compute (requires knowing p₁
everywhere). But we can define a **conditional** vector field given a specific
(x₀, x₁) pair, and average over pairs.

**Conditional path:** given (x₀, x₁), define the straight-line path:

    z_t = (1 − t)·x₀ + t·x₁,    t ∈ [0, 1]

The velocity along this path is constant:

    u_t(z_t | x₀, x₁) = x₁ − x₀

**The flow matching loss** (Lipman et al. 2022):

    L(θ) = E_{t ~ U[0,1]} E_{(x₀, x₁) ~ q}  ‖ v_θ(z_t, t) − (x₁ − x₀) ‖²

where q is the **coupling** — the joint distribution over (noise, data) pairs.

**Why this works:** It can be shown (via the law of total expectation) that
minimising the conditional FM objective is equivalent to minimising the true
marginal flow matching objective. The two objectives have the same gradient.

---

## 3. The Coupling Matters

The coupling q(x₀, x₁) decides which noise point gets paired with which data point.
This affects path geometry, not the final distribution (both converge to p₁ for
large n), but critically affects **how hard the velocity field is to learn**.

**Independent coupling:** q(x₀,x₁) = p₀(x₀) · p₁(x₁)
- Sample x₀ ~ N(0,I) and x₁ ~ data independently
- Paths can cross freely: x₀ near (3,0) might pair with x₁ near (-3,0)
- The velocity field v_θ(x,t) must learn to "fan out" and "fan in" around crossings
- More complex to learn, higher variance in gradients

**OT coupling:** q(x₀,x₁) = P*(x₀,x₁) from the Sinkhorn plan
- Pair x₀ with the x₁ that minimises expected transport cost ‖x₀ − x₁‖²
- Paths are W₂-geodesics — they don't cross (Phase 5)
- Velocity field v_θ(x,t) is smoother, lower variance
- Fewer ODE steps needed at inference

```
Text diagram — path crossings:

Independent coupling:          OT coupling:
  x₀ ──────╲────────► x₁        x₀ ──────────────► x₁
             ╲                   
  x₀' ───────╲──────► x₁'       x₀' ─────────────► x₁'
               crossing!         no crossing!
```

**Quantitatively:** OT-coupled FM has lower "kinetic energy" (mean squared velocity
integrated over time) than any other coupling. This follows directly from the
definition of W₂: the OT plan minimises Σ ‖x₀ − x₁‖².

---

## 4. Mini-Batch OT Coupling

We can't compute the exact OT plan over the full dataset at each training step.
Instead, for each training batch:

1. Sample x₀ ~ p₀ (batch_size noise vectors)   # (n, d)
2. Sample x₁ ~ p₁ (batch_size data vectors)    # (n, d)
3. Build cost matrix C_ij = ‖x₀_i − x₁_j‖²    # (n, n)
4. Run Sinkhorn(uniform_a, uniform_b, C, ε)     # (n, n) plan
5. For each row i: sample j ~ P[i,:] / P[i,:].sum()
6. Get paired batch (x₀_i, x₁_j)               # (n, d) each

This is the "mini-batch OT" approach (Tong et al. 2023 "Improving and Generalising
Flow Matching"). It gives unbiased gradient estimates and converges to the correct
OT coupling as batch_size → ∞.

---

## 5. The Training Algorithm

```
Input:  data distribution p₁, source p₀ = N(0, I)
        neural net v_θ: (ℝᵈ × [0,1]) → ℝᵈ

for step = 1, ..., T:
    x₀ = randn(batch_size, d)           sample noise
    x₁ = sample_data(batch_size)        sample real data
    
    # Mini-batch OT coupling
    C = ‖x₀_i − x₁_j‖² for all (i,j)  # (n,n) cost matrix
    P = sinkhorn(1/n, 1/n, C, ε)        # (n,n) transport plan
    j = [sample j from row i of P]      # (n,) paired indices
    x₁ = x₁[j]                          # reorder: x₁[i] is now paired with x₀[i]
    
    # Interpolate
    t  = rand(batch_size, 1)             t ~ U[0,1]
    z_t = (1-t)·x₀ + t·x₁               straight-line path
    
    # Regress onto constant velocity
    v_target = x₁ - x₀                  constant velocity along path
    v_pred   = v_θ(z_t, t)              network prediction
    loss = ‖v_pred - v_target‖²          MSE

    loss.backward(); optimiser.step()
```

---

## 6. Inference: Euler ODE Integration

After training, generate new samples:

```
x = randn(n_samples, d)              # start from noise

for t in linspace(0, 1, n_steps):
    dt = 1 / n_steps
    v = v_θ(x, t)                    # predicted velocity at (x, t)
    x = x + dt * v                   # Euler step

return x                             # ≈ samples from p₁
```

With OT coupling, paths are straight, so the Euler integrator works well with
as few as 10-20 steps. Without OT, you might need 100+ steps.

---

## 7. Architecture: The Velocity MLP

Input: [x_t (d=2), sinusoidal_emb(t) (emb_dim=16)] → (d + emb_dim)
Hidden: 3 × Linear(hidden_dim=256) + SiLU
Output: Linear → (d=2) velocity

**Sinusoidal time embedding:**

    emb(t) = [sin(t/10000^{0/dim}), sin(t/10000^{2/dim}), ...,
              cos(t/10000^{0/dim}), cos(t/10000^{2/dim}), ...]

This gives t a continuous, high-frequency representation that the MLP can
condition on precisely. Same idea as positional encoding in transformers.

**Why SiLU?** The velocity field is a smooth function of (x, t). SiLU (x·σ(x))
is smooth and non-monotone, better at fitting smooth curved velocity fields than
ReLU (piecewise linear).

---

## 8. The OT Connection — All Six Phases

| Phase | Tool | Role in Flow Matching |
|---|---|---|
| 1 (LP) | Exact OT | Conceptual baseline: what the coupling converges to |
| 2 (Duality) | Dual potentials | ∇u(x₀) IS the optimal initial velocity |
| 3 (Sinkhorn) | Regularised OT | Computes the mini-batch coupling at each step |
| 4 (Wasserstein) | W₂ distance | OT coupling minimises Σ‖x₀−x₁‖² = W₂²(p₀_batch, p₁_batch) |
| 5 (McCann) | Interpolation | z_t = (1-t)x₀ + tx₁ IS the displacement interpolant |
| 6 (Flow matching) | All of the above | Regression onto the OT-interpolated velocity field |
