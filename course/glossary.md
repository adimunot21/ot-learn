# Glossary — Optimal Transport Terminology

*Definitions in plain English, ordered by when they appear in the course.*

---

## Transport Plan (or Coupling)

A matrix P where P[i,j] tells you how much mass flows from source point i to target point j.

Think of it as a routing table: row i shows where source i's mass goes, column j shows where target j's mass comes from.

**Constraints:** rows must sum to the source distribution, columns must sum to the target distribution, all entries ≥ 0.

**Also called:** transport matrix, joint plan, coupling, joint distribution.

---

## Marginals

The row sums and column sums of a transport plan.

If P[i,j] is the transport plan, then:
- Row marginals: P[i,:].sum() = a[i] (how much source i sends, in total)
- Column marginals: P[:,j].sum() = b[j] (how much target j receives, in total)

A valid transport plan has marginals that exactly match the source and target distributions.

---

## Monge Map

A rigid assignment: each source point is sent to exactly one target point. No splitting allowed.

Think of it as a function T: every source point i has one destination T(i).

**Problem:** sometimes no Monge map exists. If you have 2 source points and 3 target points, you can't send each source point to exactly one target without splitting.

The Kantorovich relaxation (transport plan) fixes this by allowing mass to split.

---

## Kantorovich Relaxation

The generalisation of the Monge problem that allows mass to split.

Instead of a rigid function T, we use a transport plan P where each P[i,j] says how much of source i goes to target j. Source i can send some mass to target 3 and some to target 7.

This relaxation is what makes OT solvable as a linear program.

---

## Optimal Transport Plan / OT Plan

The transport plan P* that minimises total cost `Σ_{i,j} P[i,j] · C[i,j]`.

Among all valid transport plans, this one moves mass most efficiently.

---

## Earth Mover's Distance (EMD)

The minimum work (mass times distance) needed to transform one distribution into another.

Same as Wasserstein-1 distance. The name comes from the image of moving piles of earth.

---

## Wasserstein Distance (Wₚ)

A family of distances between probability distributions, defined via OT.

The Wasserstein-p distance:
```
Wₚ(μ, ν) = (min over transport plans P of: Σ_{i,j} P[i,j] · ‖xᵢ - yⱼ‖^p)^(1/p)
```

- **W₁**: minimum total transport cost (Earth Mover's Distance)
- **W₂**: minimum total squared-distance transport cost (most common in ML)

**Key property:** W₂ between two Gaussians has a closed-form formula (the Bures metric).

---

## Dual Variables / Kantorovich Potentials

Two vectors u (for the source) and v (for the target) that are the solution to the *dual* OT problem.

**Interpretation:** u[i] is the "shadow price" of having one unit of mass at source i. v[j] is the value of one unit arriving at target j.

**Constraint:** u[i] + v[j] ≤ C[i,j] for all pairs. You can't extract more value than the transport cost.

**Strong duality:** the maximum of `aᵀu + bᵀv` equals the minimum transport cost.

**Also called:** Kantorovich potentials, dual potentials.

---

## c-Transform

An operation that, given one dual variable, computes the tightest valid other dual variable.

Given u, the c-transform computes the largest v such that u[i] + v[j] ≤ C[i,j] for all i:
```
v[j] = min_i ( C[i,j] - u[i] )
```

The c-transform is used to implement the dual via alternating updates.

---

## Complementary Slackness

The condition that tells you where mass flows in the optimal plan.

**Rule:** if P*[i,j] > 0 (mass flows from i to j), then the dual variables must satisfy u[i] + v[j] = C[i,j] (tight constraint, no slack).

**Interpretation:** mass only flows along edges where the price exactly equals the transport cost. If an edge is cheaper than its price, no mass flows there.

---

## Entropy Regularisation

Adding a penalty term `ε · KL(P ‖ a⊗b)` to the OT objective.

This makes the problem strictly convex (unique solution) and allows solving via simple iterative updates (Sinkhorn). The parameter ε controls the trade-off: ε→0 recovers the exact OT plan, larger ε produces a "smoother" plan.

---

## Sinkhorn Algorithm

An iterative algorithm to solve entropy-regularised OT.

Alternates between two simple updates (rescaling rows, then columns) until the transport plan P satisfies both marginal constraints. Converges exponentially fast.

The log-domain version works in log space to avoid numerical overflow for small ε.

---

## Gibbs Kernel

The matrix `K[i,j] = exp(-C[i,j] / ε)` used in the Sinkhorn algorithm.

The optimal entropy-regularised transport plan has the form `P* = diag(f) · K · diag(g)` for some scaling vectors f and g.

---

## Scaling Vectors

The vectors f and g (or their log-equivalents u and v) used in the Sinkhorn decomposition `P* = diag(f) K diag(g)`.

Sinkhorn alternately updates: `f ← a / (Kg)` and `g ← b / (Kᵀf)`. The plan converges to P* when both marginals are satisfied.

---

## Bures Metric

The closed-form W₂ distance between two Gaussian distributions with means μ₁, μ₂ and covariance matrices Σ₁, Σ₂:

```
W₂²(N₁, N₂) = ‖μ₁ - μ₂‖² + Tr(Σ₁ + Σ₂ - 2(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})
```

The matrix square root makes this look scary but it's just a formula involving eigendecompositions.

---

## Sliced Wasserstein Distance

An approximation to W₂ that projects distributions onto random 1D lines, computes W₂ on each line (trivial in 1D), and averages.

Much faster than exact OT but systematically underestimates W₂ by a factor of 1/√d in d dimensions. Useful when exact OT is too slow.

---

## Wasserstein Barycenter

The "average" distribution in Wasserstein space.

For distributions μ₁, ..., μₖ with weights λ₁, ..., λₖ, the barycenter minimises:
```
μ* = argmin_μ Σᵢ λᵢ W₂²(μ, μᵢ)
```

Unlike a linear average (which would just overlay the distributions), the Wasserstein barycenter produces a distribution that is geometrically intermediate — a genuine "shape in between".

---

## McCann Displacement Interpolation

The "straight-line path" between two distributions in Wasserstein space.

Given source points xᵢ paired with target points yⱼ via OT plan P, the interpolation at time t ∈ [0,1]:
```
z_t = (1 - t) · xᵢ + t · yⱼ
```

At t=0: the source distribution. At t=1: the target distribution. In between: a valid intermediate distribution. Paths don't cross (this is the W₂-geodesic property).

**This is the training path used in Flow Matching (Phase 6).**

---

## Pushforward (`_#`)

Notation for applying a map to a distribution.

If T is a function and μ is a distribution, then `T_#μ` (T-pushforward of μ) is the distribution you get by applying T to all points sampled from μ.

**Example:** if μ is a distribution of heights in metres, and T(x) = x * 100 converts to centimetres, then `T_#μ` is the distribution of heights in centimetres.

**In OT:** the displacement interpolant `μₜ = ((1−t)·id + t·T)_# μ₀` means: apply the map `x ↦ (1-t)x + t·T(x)` to every point sampled from μ₀.

---

## Conditional Flow

In Flow Matching, the vector field defined for a *specific* pair (x₀, x₁).

The conditional velocity is constant: `u_t(z | x₀, x₁) = x₁ - x₀`. This is just the direction of the straight line from x₀ to x₁.

The conditional flow is tractable; the marginal flow (averaged over all pairs) is not. The key result: training on conditional flows gives the correct gradient for the marginal objective.

---

## OT Coupling (in Flow Matching)

Using the Sinkhorn transport plan to decide which noise sample x₀ pairs with which data sample x₁.

With OT coupling, each noise point is paired with its "nearest" data point (in expectation). The training paths don't cross. The velocity field is simpler to learn.

With **independent coupling**, x₀ and x₁ are paired randomly — paths cross freely, the velocity field is much more complex.

**Quantitative impact (our experiment):** OT coupling is 61–140× "straighter" (less curved paths), training loss is ~28× lower.

---

## Mini-Batch OT

Applying the OT coupling within each training mini-batch, rather than over the full dataset.

For each mini-batch: compute cost matrix C (size 128×128), run Sinkhorn to get plan P, sample pairs from P. This gives an unbiased gradient estimate of the true coupling as batch size → ∞.

Necessary because computing OT over the full dataset at every step would be intractable.

---

## Euler Integration (ODE solver)

The simplest method for integrating an ODE dx/dt = v(x, t):

```
x_{t+dt} = x_t + dt · v(x_t, t)
```

At inference in Flow Matching: start from noise x₀ ~ N(0,I), apply many small Euler steps using the trained velocity network v_θ, arrive at a sample from the data distribution.

With OT coupling (straight paths), 50 steps is sufficient. With independent coupling, you might need 200+.

---

## Jerk (Path Straightness Metric)

In this course: the sum of squared velocity changes along a path.

```
jerk = Σ_t ‖v(t+1) - v(t)‖²
```

A perfectly straight path has zero jerk (constant velocity). A curved path has high jerk.

**In Phase 6:** OT paths have jerk ~7e-7, independent paths ~4e-5. The OT paths are ~60× straighter.
