# Phase 5 Derivation — Wasserstein Barycenters & Interpolation

## Motivation

We have a distance (W₂) between distributions. Now we can ask the natural follow-up:
given a collection of distributions, what is their **average**?

In Euclidean space the average of points x₁,…,xₖ with weights λ₁,…,λₖ is defined
as the Fréchet mean — the point that minimises weighted squared distances:

    x* = argmin_x Σᵢ λᵢ ‖x − xᵢ‖²     → x* = Σᵢ λᵢ xᵢ  (linear average)

Replacing ‖·‖² with W₂² gives the **Wasserstein barycenter**:

    μ* = argmin_μ Σᵢ λᵢ W₂²(μ, μᵢ)

There's no closed-form Σᵢ λᵢ μᵢ for distributions — distributions can't be averaged
by linear combination (that doesn't respect their geometry).
The Wasserstein barycenter respects the geometry.

---

## 1. McCann Interpolation (2 distributions)

Before tackling barycenters of k distributions, understand the k=2 case.

Given μ₀ and μ₁, define the **displacement interpolant** at time t ∈ [0,1]:

    μₜ = ((1−t)·id + t·T)_# μ₀

where T is the optimal transport map from μ₀ to μ₁ (pushes μ₀ forward to μ₁),
and the pushforward ((1−t)·id + t·T)_# means: move each sample xᵢ from μ₀ to
the intermediate point (1−t)xᵢ + t·T(xᵢ).

**In pictures:**
```
t=0.0:  μ₀ = {x₁, x₂, x₃}     (starting cloud)
t=0.5:  μ₀.₅ = {(x₁+T(x₁))/2, ...}  (midpoint cloud)
t=1.0:  μ₁ = {T(x₁), T(x₂), T(x₃)}  (target cloud)
```

Each point travels in a **straight line** from xᵢ to T(xᵢ). The displacement
interpolant is literally straight-line interpolation along optimal transport paths.

**Why this is the "correct" path:**
μₜ is the W₂-geodesic from μ₀ to μ₁. It minimises total "kinetic energy" (speed²
integrated over time). Any other path between μ₀ and μ₁ has higher total cost.

**For discrete distributions** (samples, not maps):
Replace the map T with the transport plan P ∈ ℝⁿˣᵐ.
Each unit of mass Pᵢⱼ travels from xᵢ to yⱼ along a straight line.
At time t, that mass sits at (1−t)xᵢ + tyⱼ.

    Interpolated cloud at time t:
    For each (i,j) pair with Pᵢⱼ > 0: add point z = (1−t)xᵢ + tyⱼ with weight Pᵢⱼ

---

## 2. The Barycenter Fixed-Point Equation

For k distributions μ₁,…,μₖ with weights λᵢ ≥ 0, Σλᵢ = 1:

    μ* = argmin_μ Σᵢ λᵢ W₂²(μ, μᵢ)

By the envelope theorem (differentiating through the minimum), the gradient of
W₂²(μ, μᵢ) with respect to the support of μ at a point xⱼ is:

    ∂/∂xⱼ  W₂²(μ, μᵢ) = 2 (xⱼ − Tᵢ(xⱼ))

where Tᵢ(xⱼ) is the barycentre of all mass transported from xⱼ to μᵢ:

    Tᵢ(xⱼ) = Σₗ (Pᵢ[j,l] / μ(xⱼ)) · yᵢ_l      ("barycentric projection")

Setting the gradient to zero:

    Σᵢ λᵢ (xⱼ − Tᵢ(xⱼ)) = 0
    xⱼ = Σᵢ λᵢ Tᵢ(xⱼ)
    xⱼ = Σᵢ λᵢ Σₗ (Pᵢ[j,l] / pⱼ) yᵢ_l             ...(★)

This is the **fixed-point equation** for the barycenter.
Each barycenter point xⱼ must equal the weighted average of where it gets transported
to in each reference measure. Nothing moves if this condition is satisfied.

---

## 3. The Iterative Algorithm

Equation (★) suggests a simple fixed-point iteration (Alvarez-Esteban et al. 2016):

```
Initialize:  x = random or average of all yᵢ's,  p = uniform weights

Repeat:
  (1) For each reference measure i:
        Cᵢ = pairwise_sq_dist(x, yᵢ)        # (n, mᵢ)  cost matrix
        Pᵢ = Sinkhorn(p, bᵢ, Cᵢ, ε)         # (n, mᵢ)  transport plan

  (2) Update support points:
        xⱼ ← Σᵢ λᵢ · Σₗ (Pᵢ[j,l] / pⱼ) · yᵢ_l    for all j
           = Σᵢ λᵢ · (Pᵢ[j,:] / pⱼ) @ yᵢ            matrix form

Until positions x stop changing.
```

Step 2 in matrix form:

    x ← Σᵢ λᵢ · diag(1/p) · Pᵢ · yᵢ

Shape trace:
    Pᵢ           : (n, mᵢ)
    diag(1/p)    : effectively (n,) inverse weights
    Pᵢ @ yᵢ     : (n, mᵢ) @ (mᵢ, d) = (n, d)
    Pᵢ @ yᵢ / p[:,None] : (n, d)  barycentric projection
    x_new        : (n, d)  weighted sum over all i

---

## 4. Why It Converges

The algorithm minimises the objective Σᵢ λᵢ W₂²(μ, μᵢ) via coordinate descent:
- Step 1 holds x fixed and minimises over P (the optimal transport plans) → Sinkhorn
- Step 2 holds P fixed and minimises over x (the support positions) → closed-form

This alternating minimisation is guaranteed to decrease the objective at every step.
For the regularised problem (with ε > 0), the objective is strictly convex in (x, P)
jointly, so the iteration converges to a unique solution.

---

## 5. Worked Example — 2D Point Clouds

Consider k=2, λ₁=λ₂=0.5:

    μ₁: 4 points on a circle   {(1,0), (0,1), (-1,0), (0,-1)}   equal weights
    μ₂: 4 points on a square   {(1,1), (-1,1), (-1,-1), (1,-1)} equal weights

Iteration 1 starting from x = average of all points = (0,0) for each of 4 points
(all at origin):

    C₁ = pairwise dist²(x, y₁) — all equal since x is all-(0,0)
    P₁ = uniform transport plan (Sinkhorn with flat cost)

    After transport update: x ← 0.5 * T₁(x) + 0.5 * T₂(x)

The barycenter will converge to points "between" the circle and square —
roughly at the corners of a rounded square with radius ~√2.

For t=0.5 McCann interpolation:
    Pair each circle point with its nearest square point.
    At t=0.5: each point is halfway between.

    (1,0) ↔ (1,1): midpoint (1, 0.5)
    (0,1) ↔ (-1,1): midpoint (-0.5, 1)
    etc.

---

## 6. Connection to Phase 6 (Flow Matching)

This is where everything connects.

Flow matching (Phase 6) trains a neural network to transport samples from a simple
source distribution (Gaussian noise) to a complex target distribution (images, data).

The key ingredient: for each noise sample xᵢ ~ N(0,I) and data sample yⱼ, define
a "flow path": xᵢ + t(yⱼ − xᵢ) = (1−t)xᵢ + tyⱼ.

**This is exactly McCann interpolation** with cost matrix Cᵢⱼ = ‖xᵢ − yⱼ‖².

The OT coupling (using Sinkhorn) decides which noise sample gets paired with which
data point — just like the transport plan Pᵢⱼ decides which source mass goes to
which target. Flow matching with OT coupling uses our exact Sinkhorn solver to
construct the straightest possible paths, minimising path crossings.

---

## 7. Key Takeaways

| Concept | What it means |
|---|---|
| Fréchet mean in W₂ | minimise Σᵢ λᵢ W₂²(μ, μᵢ) over μ |
| McCann interpolant | straight-line transport: z_t = (1−t)x + tT(x) |
| Geodesic | McCann interpolation IS the W₂-geodesic |
| Fixed-point equation | xⱼ = Σᵢ λᵢ Tᵢ(xⱼ) — each point = weighted bary-proj |
| Barycentric projection | Tᵢ(xⱼ) = (Pᵢ[j,:]/pⱼ) @ yᵢ — conditional mean |
| OT coupling in FM | same Sinkhorn plan defines "straight" flow paths |
