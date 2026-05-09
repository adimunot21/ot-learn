# Chapter 4 — Wasserstein Distances

## The Problem and Why It Matters

We can now compute optimal transport plans. But often we don't care about the plan —
we just want to know **how far apart two distributions are**.

The **Wasserstein distance** is the OT cost at the optimum: the minimum total work
needed to transform one distribution into the other.

Why does this matter?

**The key failure of KL and JS divergence.** Consider two point masses:

```
μ = δ₀   (all mass at x = 0)
ν_t = δ_t  (all mass at x = t)
```

These are clearly "distance t apart" in any intuitive sense.

```
KL(μ ‖ ν_t)  = ∞       for all t ≠ 0   (undefined when supports don't overlap)
JS(μ, ν_t)   = log 2   for all t ≠ 0   (constant — no gradient signal!)
W₁(μ, ν_t)  = t                         (reflects the geometry)
```

This is exactly why WGANs use Wasserstein distance: when the generator produces
distributions with disjoint support from real data (common early in training),
KL and JS give zero gradient — training stalls. Wasserstein always provides a
useful signal proportional to how far off the generator is.

---

## Definition

The **Wasserstein-p distance** between distributions μ and ν on a metric space:

```
W_p(μ, ν)^p  =  min_{P ∈ U(μ,ν)}  ∫∫ d(x, y)^p  dP(x, y)
```

where d is the ground metric (typically Euclidean) and U(μ,ν) is the set of all
joint distributions with marginals μ and ν (same transport polytope from Phase 1).

For discrete distributions this is exactly the LP from Phase 1, with cost matrix
Cᵢⱼ = d(xᵢ, yⱼ)^p.

**W_p itself** (not W_p^p) satisfies the triangle inequality and is a proper metric.
The p-th power is a convention that makes the scaling work out.

---

## Metric Properties

W_p is a **metric** on the space of probability measures with finite p-th moment.

```
W_p(μ,ν) = 0   ⟺   μ = ν               (identity of indiscernibles)
W_p(μ,ν) = W_p(ν,μ)                     (symmetry — transport cost is same both ways)
W_p(μ,σ) ≤ W_p(μ,ν) + W_p(ν,σ)        (triangle inequality)
```

The triangle inequality comes from "gluing" two transport plans via a shared
intermediate measure ν. It's stated here without full proof (the proof uses the
disintegration theorem).

The space of probability measures equipped with W₂ is called **Wasserstein space**.
It's a rich metric space where we can define geodesics (straight-line paths between
measures), barycenters (weighted averages), and gradients. Phase 5 builds on this.

---

## 1D Closed Form: The Quantile Formula

In one dimension, we don't need Sinkhorn or LP at all.

**Quick review:** The **CDF** (cumulative distribution function) F_μ(x) is the probability that a sample from μ lands below x — a non-decreasing function from 0 to 1. The **quantile function** F⁻¹(t) is its inverse: the value x such that F_μ(x) = t. For equal-weight samples, it's just the sorted array. (Full review in `prerequisites.md`, section 10.)

**Theorem.** For distributions μ, ν on ℝ with CDFs F_μ and F_ν:

```
W_p(μ, ν)^p  =  ∫₀¹ |F_μ⁻¹(t) − F_ν⁻¹(t)|^p  dt
```

where F⁻¹(t) = inf{x : F(x) ≥ t} is the quantile function (inverse CDF).

**Why?** In 1D, the optimal transport map is always monotone non-decreasing — mass
always moves "in order". The cheapest thing to do is match the t-th quantile of μ
to the t-th quantile of ν. No mass ever crosses.

```
Text diagram — why quantile matching is optimal in 1D:

Source:  [x₁  x₂  x₃  x₄]   sorted: [1   3   5   7]
Target:  [y₁  y₂  y₃  y₄]   sorted: [2   4   6   8]

Optimal:  1→2, 3→4, 5→6, 7→8    (match by rank, cost = 4×1 = 4)

Sub-optimal: 1→4, 3→2, 5→6, 7→8 (crossing paths, cost = 3+1+1+1 = 6 > 4)
```

Any plan that "crosses" (sends lower mass to a higher target than another mass)
can be uncrossed at no greater cost. So the optimal plan never crosses.

**For equal-weight samples**, the formula reduces to sorting and pairing:

```
W_p(μ, ν)^p  =  (1/n) Σᵢ |x_sort[i] − y_sort[i]|^p
```

Complexity: O(n log n) dominated by sorting.

**Worked example (from derive.md):**

```
μ: uniform on {1, 3, 5, 7}   →   sorted: [1, 3, 5, 7]
ν: uniform on {2, 4, 6, 8}   →   sorted: [2, 4, 6, 8]

W₁ = (1/4)(|1−2| + |3−4| + |5−6| + |7−8|) = (1/4)(1+1+1+1) = 1.0
```

Geometric sense: each point shifts exactly 1 unit to the right.

---

## W₂ Between Gaussians: Closed Form

For μ = N(m₁, Σ₁) and ν = N(m₂, Σ₂) in ℝᵈ (Dowson & Landau, 1982):

```
W₂(μ,ν)² = ‖m₁ − m₂‖²  +  B(Σ₁, Σ₂)²

B(Σ₁,Σ₂)² = Tr(Σ₁) + Tr(Σ₂) − 2·Tr( (Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2} )
```

The term B is the **Bures metric** on positive definite matrices.

**For univariate Gaussians** N(μ₁, σ₁²) and N(μ₂, σ₂²):

```
W₂² = (μ₁ − μ₂)²  +  (σ₁ − σ₂)²
```

It's just the squared Euclidean distance between their (mean, std) parameters.

**Worked example:**

```
μ = N(0, 1),   ν = N(3, 4)   [i.e. mean=3, variance=4, std=2]

W₂² = (0 − 3)² + (1 − 2)² = 9 + 1 = 10
W₂  = √10 ≈ 3.162
```

Our closed-form code confirms: `w2_gaussian([0],[1²],[3],[2²]) = 3.162278`. ✓

---

## W₂ via Sinkhorn (Any Dimension)

When distributions aren't Gaussian and live in ℝᵈ with d > 1, we fall back to
Sinkhorn with the squared-Euclidean cost matrix:

```
C_ij = ‖xᵢ − yⱼ‖²    (n, m)

Efficient computation:
‖xᵢ−yⱼ‖² = ‖xᵢ‖² − 2 xᵢ·yⱼ + ‖yⱼ‖²
           = x_sq[i] + y_sq[j] − 2 (X @ Y.T)[i,j]
```

Shape trace:
```
x_sq = sum(x**2, axis=1)[:,None]    (n, 1)
y_sq = sum(y**2, axis=1)[None,:]    (1, m)
cross = x @ y.T                     (n, m)
C = x_sq + y_sq - 2*cross           (n, m)   ← broadcasting
```

Then Sinkhorn gives the regularized OT cost, and `sqrt(cost)` approximates W₂.

From Test 4: for n=200 samples from N(0,1) vs N(3,4), Sinkhorn gives W₂ ≈ 3.29
against the exact 3.16 — error 0.13, which shrinks as n grows and ε → 0.

---

## Sliced Wasserstein Distance

Exact W₂ via Sinkhorn costs O(n²) in memory (storing C) and O(n² · T) in time.
For n=10,000 samples in d=128 dimensions — forget it.

**The sliced trick:** reduce to 1D where computing W_p costs O(n log n).

Project both distributions onto a random unit vector θ ∈ Sᵈ⁻¹:

```
x_proj = X @ θ     (n,)
y_proj = Y @ θ     (m,)
```

Compute 1D W_p on the projections (sort and pair). Average over L random directions:

```
SW_p(μ,ν)  ≈  (1/L  Σ_{l=1}^{L}  W_p^p(x @ θ_l, y @ θ_l))^{1/p}
```

**Complexity:** O(L · n log n · d) — linear in n (after projections).

### The 1/√d underestimation

Sliced Wasserstein is **not equal to W_p**. For unit-covariance Gaussians shifted by μ in ℝᵈ:

```
Projection onto θ:  W₂(proj) = |θᵀμ|
Average over θ:     E[|θᵀμ|²] = ‖μ‖²/d      (equal contribution per dimension)
SW₂ = ‖μ‖/√d = W₂/√d
```

In 2D (d=2), SW₂ ≈ W₂/√2 ≈ 0.707 · W₂.

From Test 5:
```
True W₂ = √5 ≈ 2.236    True SW₂ ≈ W₂/√2 ≈ 1.581
Sinkhorn W₂ = 2.241     Sliced W₂ = 1.569   (matches theory to 0.012)
```

The sliced distance isn't a bad approximation of W₂ — **it's answering a different
question**: average 1D transport cost over all projections. For high-dimensional data
(d >> 1), it's often the only tractable option, and its gradients are well-defined.

---

## The Code, Line by Line

### 1D Wasserstein via quantile integration

```python
def w1_1d(x, y, a=None, b=None):
    # Sort both distributions
    x_sorted, a_sorted = x[argsort(x)], a[argsort(x)]
    y_sorted, b_sorted = y[argsort(y)], b[argsort(y)]

    # Build CDF breakpoints for each distribution
    a_cdf = concat([[0], cumsum(a_sorted)])   # (n+1,)  CDF of μ
    b_cdf = concat([[0], cumsum(b_sorted)])   # (m+1,)  CDF of ν

    # Merge into one set of quantile levels
    all_levels = unique(concat([a_cdf, b_cdf]))

    # Integrate |F_μ⁻¹(t) − F_ν⁻¹(t)| dt using midpoint rule
    w1 = sum(|quantile_μ(t_mid) − quantile_ν(t_mid)| * dt
             for consecutive (t, t+dt) in all_levels)
    return w1
```

### Sliced Wasserstein

```python
def sliced_wasserstein(x, y, p=2, n_projections=200):
    # x: (n,d),  y: (m,d)
    d = x.shape[1]

    # Sample L random unit vectors
    thetas = randn(n_projections, d)           # (L, d)
    thetas /= norm(thetas, axis=1)             # (L, d) unit vectors

    wp_values = empty(n_projections)
    for l in range(n_projections):
        x_proj = x @ thetas[l]                # (n,)  1D projection
        y_proj = y @ thetas[l]                # (m,)
        wp_values[l] = w_p_1d(x_proj, y_proj) ** p

    return mean(wp_values) ** (1/p)
```

---

## Shape Traces

| Variable | Shape | Meaning |
|---|---|---|
| `x` | `(n, d)` | source samples |
| `y` | `(m, d)` | target samples |
| `thetas` | `(L, d)` | random unit projection vectors |
| `x @ thetas[l]` | `(n,)` | 1D projections of source |
| `C = x_sq + y_sq - 2*(x@y.T)` | `(n, m)` | squared-Euclidean cost matrix |
| `wp_values` | `(L,)` | 1D W_p^p for each projection |

---

## Summary Table

| Variant | Formula | Complexity | Use when |
|---|---|---|---|
| W_p via LP | exact, Phase 1 | O((nm)³) | tiny problems, need exactness |
| W_p via Sinkhorn | regularized | O(nm · T) | medium-scale, differentiable |
| W_p in 1D | sort + integrate | O(n log n) | 1D distributions |
| W₂ Gaussian | Bures metric | O(d³) | Gaussian distributions |
| Sliced W_p | average 1D W_p | O(L·n log n·d) | high-d, fast approximation |

---

## What's Next

We have a distance. Now we can ask: given a set of distributions {μ₁,…,μₖ},
what is their **average** in Wasserstein space?

This is the Wasserstein barycenter — the distribution that minimises the weighted
sum of W₂ distances to all inputs. It's the Fréchet mean of distributions, and it
requires iterating Sinkhorn in a clever loop.

Phase 5 builds this, and the barycenter is the mathematical tool that unlocks
the interpolation in flow matching (Phase 6).
