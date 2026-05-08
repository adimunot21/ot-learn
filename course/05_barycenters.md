# Chapter 5 — Wasserstein Barycenters & McCann Interpolation

## The Problem and Why It Matters

We have a distance (W₂). A natural next question: given a collection of distributions,
what is their **average**?

In Euclidean space, the average (Fréchet mean) of points x₁,…,xₖ with weights λ is:

```
x* = argmin_x Σᵢ λᵢ ‖x − xᵢ‖²   →   x* = Σᵢ λᵢ xᵢ   (linear average)
```

Replacing ‖·‖² with W₂² gives the **Wasserstein barycenter**:

```
μ* = argmin_μ Σᵢ λᵢ W₂²(μ, μᵢ)
```

There's no formula like Σλᵢμᵢ for distributions — that would mix masses linearly,
ignoring geometry. The Wasserstein barycenter respects geometry.

**Why this matters:**
- Averaging shapes (images, point clouds) without blurring
- Interpolating between distributions in a geometrically meaningful way
- The foundation of flow matching: the optimal interpolation between noise and data
  is the W₂ geodesic — the McCann interpolant

---

## McCann Displacement Interpolation (k=2)

Before averaging k distributions, understand the path between 2.

Given μ₀ and μ₁ with optimal transport map T (the Monge map from Phase 1),
the **displacement interpolant** at time t ∈ [0,1]:

```
μₜ = ((1−t)·id + t·T)_# μ₀
```

Each point xᵢ in μ₀ travels in a **straight line** to T(xᵢ) in μ₁.
At time t, that point sits at (1−t)xᵢ + t·T(xᵢ).

```
Text diagram — straight-line paths along OT:

  t=0.0    t=0.25   t=0.5    t=0.75   t=1.0
  ●                  ○                   ■
   \                / \                 /
    ●──────────────○   ○───────────────■
   /                \ /                 \
  ●                  ○                   ■

Source (●) travels to target (■) along straight lines.
No paths cross — the OT plan ensures this.
```

This is the **W₂-geodesic** — the "straight line" in Wasserstein space from μ₀ to μ₁.
It minimises total kinetic energy ∫₀¹ ‖velocity‖² dt over all paths.

**For discrete distributions** (point clouds):
We don't have the Monge map, but we have the transport plan P (from Sinkhorn).
For each active pair (i,j) with P_ij > 0, place mass P_ij at the interpolated point:

```
z_ij(t) = (1−t)·xᵢ + t·yⱼ     weight: P_ij
```

At t=0: all points collapse to their source positions (z_ij(0) = xᵢ for all j).
At t=1: all points arrive at their target positions (z_ij(1) = yⱼ for all i).

Note: because Sinkhorn makes all P_ij > 0, the t=0 frame technically has n×m points
(xᵢ appearing once per pairing). Their weighted mean exactly recovers the source
distribution — the correct check is always at the distribution level.

---

## The Barycenter Fixed-Point Equation

For k distributions μ₁,…,μₖ with weights λᵢ, the barycenter satisfies:

```
μ* = argmin_μ Σᵢ λᵢ W₂²(μ, μᵢ)
```

Differentiate with respect to the support of μ at point xⱼ.
The gradient of W₂²(μ, μᵢ) w.r.t. xⱼ is:

```
∂/∂xⱼ W₂²(μ, μᵢ) = 2 (xⱼ − Tᵢ(xⱼ))
```

where Tᵢ(xⱼ) is the **barycentric projection** — the conditional mean of where μ
sends mass from xⱼ into μᵢ:

```
Tᵢ(xⱼ) = Σₗ (Pᵢ[j,l] / pⱼ) · yᵢ_l      ← weighted average of targets
```

Setting the total gradient to zero:

```
Σᵢ λᵢ · 2(xⱼ − Tᵢ(xⱼ)) = 0
xⱼ = Σᵢ λᵢ · Tᵢ(xⱼ)                       ...(★)
```

**Each barycenter point must equal the weighted average of where it gets transported
to in each reference measure.** If any point is "off", it moves toward the average
of its images.

---

## The Algorithm

Equation (★) suggests a fixed-point iteration:

```
Initialize: x = random sample from ∪μᵢ,  p = uniform weights

Repeat:
  For each i:
    Cᵢ = pairwise_sq_dist(x, yᵢ)          # (n, mᵢ) cost matrix
    Pᵢ = Sinkhorn(p, bᵢ, Cᵢ, ε)           # (n, mᵢ) transport plan

  x ← Σᵢ λᵢ · (Pᵢ / p[:,None]) @ yᵢ      # (n, d) updated support
Until max displacement < tolerance
```

Step 2 in detail — shape trace:
```
Pᵢ                 : (n, mᵢ)
p[:, None]         : (n, 1)
Pᵢ / p[:,None]     : (n, mᵢ)    row-normalised plan (rows sum to 1)
@ yᵢ               : (n, mᵢ) @ (mᵢ, d) = (n, d)   barycentric projection
λᵢ * (...)         : (n, d)    weighted contribution of measure i
Σᵢ (...)           : (n, d)    new support positions
```

This is an **alternating minimisation**:
- Inner step (Sinkhorn): fix x, find optimal transport plans Pᵢ
- Outer step: fix Pᵢ, find optimal support positions x (closed form!)

Each outer step strictly decreases the objective. For ε > 0, the problem is strictly
convex and the algorithm converges to a unique barycenter.

---

## Worked Example: Circle + Square Barycenter

```
μ₁: 60 points on a unit circle
μ₂: 60 points on a 1.2-side square
λ = [0.5, 0.5]
```

After 30 outer iterations:

```
W₂(barycenter, circle) = 0.300
W₂(barycenter, square) = 0.318
Asymmetry              = 0.018   ← nearly equal, as expected for λ=[0.5, 0.5]
```

The barycenter sits between the two shapes — a "rounded square" or "squircle"
with intermediate geometry. Not a blur: each individual support point has moved
to a geometrically meaningful position.

For comparison, a naive linear average (Σᵢ λᵢ μᵢ) would just overlay both point
clouds — the result would look like both a circle AND a square, not something between.
The Wasserstein barycenter gives a shape that is genuinely intermediate.

---

## Three-Shape Barycenter

With k=3 equal-weight measures (circle, square, triangle):

```
W₂(bary, circle)   = 0.292
W₂(bary, square)   = 0.387
W₂(bary, triangle) = 0.349
Spread             = 0.095   ← all three distances are similar, as expected
```

The barycenter is equidistant (in W₂) from all three input shapes.

---

## The Code, Line by Line

### McCann interpolation

```python
def mccann_interpolation(x, y, transport_plan, t):
    # Find all (i,j) pairs with non-trivial mass P_ij > 0
    i_idx, j_idx = np.nonzero(transport_plan > 1e-9)   # (K,) each
    w = transport_plan[i_idx, j_idx]                    # (K,) weights

    xi = x[i_idx]   # (K, d)   source points for each active pair
    yj = y[j_idx]   # (K, d)   target points

    # Straight-line interpolation — the key line
    z = (1 - t) * xi + t * yj   # (K, d)
    return z, w / w.sum()
```

### The outer barycenter loop

```python
for outer_iter in range(max_iter):
    x_new = zeros(n_support, d)

    for i, (yi, bi, li) in enumerate(zip(measures, weights, lambdas)):
        # Cost matrix between current x and reference measure yᵢ
        Ci = pairwise_sq_dist(x, yi)             # (n_support, mᵢ)

        # Inner Sinkhorn: optimal plan from p to bᵢ
        Pi = log_sinkhorn(Ci, p, bi, ε)          # (n_support, mᵢ)

        # Barycentric projection: for each j, weighted average of targets
        Ti_x = (Pi / p[:,None]) @ yi             # (n_support, d)
        x_new += li * Ti_x

    displacement = |x_new - x|.max()
    x = x_new
    if displacement < tol: break
```

---

## Shape Traces

| Variable | Shape | Meaning |
|---|---|---|
| `x` | `(n_support, d)` | barycenter support points |
| `p` | `(n_support,)` | barycenter weights (uniform) |
| `yi` | `(mᵢ, d)` | reference measure i support |
| `Ci` | `(n_support, mᵢ)` | squared-dist cost matrix |
| `Pi` | `(n_support, mᵢ)` | Sinkhorn transport plan |
| `Pi / p[:,None]` | `(n_support, mᵢ)` | row-normalised: rows sum to 1 |
| `Pi @ yi` | `(n_support, d)` | bary projection of each x point into μᵢ |
| `x_new` | `(n_support, d)` | updated support after one outer step |

---

## Summary Table

| Concept | Definition |
|---|---|
| W₂ Fréchet mean | `argmin_μ Σᵢ λᵢ W₂²(μ, μᵢ)` |
| McCann interpolant | straight-line transport: `z_t = (1−t)x + tT(x)` |
| W₂-geodesic | McCann interpolation is the shortest path in Wasserstein space |
| Barycentric projection | `Tᵢ(xⱼ) = (Pᵢ[j,:]/pⱼ) @ yᵢ` — conditional transport mean |
| Fixed-point eq. | `xⱼ = Σᵢ λᵢ Tᵢ(xⱼ)` — barycenter points don't move |
| Alternating min. | inner Sinkhorn (fix x) + outer update (fix P) |
| Linear average | Σλᵢμᵢ overlays distributions — not geometrically meaningful |

---

## Connection to Phase 6

Flow matching (Phase 6) trains a neural network to move samples from noise N(0,I)
to a target data distribution. The "path" each noise sample takes to reach a data
point is exactly a McCann interpolation frame:

```
z(t) = (1−t)·xᵢ  +  t·yⱼ         xᵢ ~ noise,  yⱼ ~ data
```

The velocity along this path is constant: `dz/dt = yⱼ − xᵢ`.

Flow matching trains `v_θ(z, t) ≈ yⱼ − xᵢ` — a network that predicts this velocity.
The OT coupling (using Sinkhorn to pair xᵢ with yⱼ) ensures paths don't cross,
making the velocity field simpler to learn.

**The entire flow matching algorithm is McCann interpolation + a regression network.**

## What's Next

Phase 6: we implement flow matching from scratch. We'll use Sinkhorn to compute the
OT coupling between noise and data batches (giving straight paths), train an MLP
velocity network to predict those straight-line velocities, and solve the ODE at
inference time using Euler integration. The payoff: a generative model where
everything we've built — LP, duality, Sinkhorn, Wasserstein, barycenters — connects.
