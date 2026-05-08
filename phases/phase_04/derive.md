# Phase 4 Derivation — Wasserstein Distances

## From Transport Plans to a Distance

In Phases 1–3 we computed the optimal transport plan P*.
The **Wasserstein distance** is just the scalar OT cost at that optimum — we throw
away the plan and keep only the number.

    W_p(μ, ν)^p  =  min_{P ∈ U(μ,ν)}  Σᵢⱼ d(xᵢ, yⱼ)^p · Pᵢⱼ

where d is the ground metric (usually Euclidean distance).

For p=1: W₁(μ,ν) = min cost using distance as cost.
For p=2: W₂(μ,ν)² = min cost using squared distance. W₂ itself is the square root.

The p-th power is a convention so that W_p satisfies the triangle inequality and
scales linearly with the ground metric (not like its p-th power).

---

## 1. Why Wasserstein is Better Than KL or JS Divergence

Consider two unit masses:
    μ = δ_0    (all mass at 0)
    ν_t = δ_t  (all mass at t)

**KL divergence:** KL(μ‖ν_t) = ∞ for any t ≠ 0 (disjoint supports).
**JS divergence:** JS(μ, ν_t) = log 2 for any t ≠ 0 (constant — no gradient!).
**W₁(μ, ν_t):** = t (linear, smooth, reflects the geometry).

This is why WGANs use Wasserstein distance as a training signal: when generated and
real distributions are disjoint (common early in training), KL/JS give no gradient.
Wasserstein always gives a useful signal proportional to the actual gap.

---

## 2. Metric Properties

W_p is a metric on the space of probability distributions with finite p-th moment.

**Proof that W_p > 0 for μ ≠ ν:**
If μ ≠ ν, any P ∈ U(μ,ν) must transport some mass across a non-zero distance,
so ⟨C, P⟩ > 0.

**Proof of triangle inequality:**
Given three measures μ, ν, σ and optimal plans P_μν, P_νσ, we can "glue" them
via the common marginal ν to construct a plan P_μσ whose cost is bounded by
W_p(μ,ν) + W_p(ν,σ). (The gluing lemma — stated here, proof is measure-theoretic.)

The space of probability measures equipped with W₂ is called **Wasserstein space**.
It's a metric space with rich geometric structure — geodesics, barycenters, gradients
are all well-defined. Phases 5 and 6 exploit this.

---

## 3. 1D Closed Form — The Quantile Formula

In 1D (d=1), W_p has a remarkably clean closed form that requires no LP or Sinkhorn.

**Theorem:** For μ, ν on ℝ with CDFs F_μ, F_ν:

    W_p(μ, ν)^p  =  ∫₀¹ |F_μ⁻¹(t) − F_ν⁻¹(t)|^p dt

where F⁻¹(t) = inf{x : F(x) ≥ t} is the quantile function (inverse CDF).

**Proof sketch:**
In 1D, the optimal transport map T is monotone non-decreasing — it always "matches"
mass in order (the cheapest thing to do). The monotone map is T = F_ν⁻¹ ∘ F_μ.
Substituting u = F_μ(x):

    W_p^p = ∫ |x − T(x)|^p dμ(x)
           = ∫₀¹ |F_μ⁻¹(u) − F_ν⁻¹(u)|^p du

**For discrete distributions** with sorted weights:
Sort both distributions. Compute the cumulative weights (the empirical CDF).
Interpolate the quantile function. Integrate numerically.

**Special case — equal sample sizes:**
If both μ and ν have n equal-weight samples (aᵢ = bⱼ = 1/n):

    W_p(μ,ν)^p  =  (1/n) Σᵢ |x_sort[i] − y_sort[i]|^p

Just sort both, pair them up, compute average p-th power of differences.
This is O(n log n) (dominated by sorting) vs O(n³) for the LP.

**Worked example (W₁, n=4):**

    μ: masses at {1, 3, 5, 7} with equal weights (1/4 each)
    ν: masses at {2, 4, 6, 8} with equal weights (1/4 each)

    Sort both: already sorted.
    Pairings: (1→2), (3→4), (5→6), (7→8)
    W₁ = (1/4)(|1-2| + |3-4| + |5-6| + |7-8|) = (1/4)(1+1+1+1) = 1.0

This makes geometric sense: each point moves exactly 1 unit to the right.

---

## 4. W₂ Between Gaussians — Closed Form

For μ = N(m₁, Σ₁) and ν = N(m₂, Σ₂) in ℝᵈ, W₂ has a closed form (Dowson & Landau 1982):

    W₂²(μ, ν)  =  ‖m₁ − m₂‖²  +  B(Σ₁, Σ₂)²

where B(Σ₁, Σ₂) is the **Bures metric**:

    B(Σ₁, Σ₂)² = Tr(Σ₁) + Tr(Σ₂) − 2 Tr((Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})

**For univariate Gaussians** N(μ₁, σ₁²) and N(μ₂, σ₂²):

    W₂²(μ, ν)  =  (μ₁ − μ₂)²  +  (σ₁ − σ₂)²

Just the squared Euclidean distance between their (mean, std) parameters.

**Worked example:**
    μ₁ = N(0, 1),   μ₂ = N(3, 2)
    W₂² = (0−3)² + (1−2)² = 9 + 1 = 10
    W₂  = √10 ≈ 3.162

This will be our ground truth for validating the Sinkhorn approximation in 1D.

---

## 5. Sliced Wasserstein Distance

For d-dimensional distributions, exact W₂ requires Sinkhorn with O(n²) cost matrix —
expensive for large n.

**Key idea:** reduce to 1D where the closed form is free.

Project both distributions onto a random unit vector θ ∈ Sᵈ⁻¹:

    X_θ = {θᵀxᵢ}    (projected source points)
    Y_θ = {θᵀyⱼ}    (projected target points)

Compute the 1D Wasserstein distance on the projections.
Average over many random directions:

    SW_p(μ, ν)  =  𝔼_{θ ~ Uniform(Sᵈ⁻¹)} [W_p(θ_#μ, θ_#ν)^p]^{1/p}

In practice, approximate the expectation with L random directions:

    SW_p(μ, ν)  ≈  (1/L Σ_{l=1}^{L} W_p^p(proj_l(μ), proj_l(ν)))^{1/p}

**Each 1D W_p is O(n log n) via sorting.**
**Total complexity: O(L · n log n · d)** for the projections.

**Worked example (d=2, L=3 directions):**

    μ: 4 points at [(0,0), (1,0), (0,1), (1,1)]  uniform weights
    ν: 4 points at [(2,0), (3,0), (2,1), (3,1)]  uniform weights
    (ν is μ shifted right by 2)

    True W₁: should be 2 (each point shifts right by 2).

    Direction θ₁ = [1, 0]  (x-axis):
        projected μ: {0, 1, 0, 1} → sorted: [0, 0, 1, 1]
        projected ν: {2, 3, 2, 3} → sorted: [2, 2, 3, 3]
        W₁ = (1/4)(|0-2| + |0-2| + |1-3| + |1-3|) = (1/4)(2+2+2+2) = 2.0

    Direction θ₂ = [0, 1]  (y-axis):
        projected μ: {0, 0, 1, 1} → sorted: [0, 0, 1, 1]
        projected ν: {0, 0, 1, 1} → sorted: [0, 0, 1, 1]
        W₁ = 0.0  (no difference in y-direction)

    Direction θ₃ = [1/√2, 1/√2]  (diagonal):
        projected μ: {0, 1/√2, 1/√2, √2} → sorted: [0, .707, .707, 1.414]
        projected ν: {2/√2, 3/√2, 2/√2, 3/√2+1/√2} → hmm... [1.414, 2.121, 1.414, 2.828]
                   → sorted: [1.414, 1.414, 2.121, 2.828]

    SW₁ ≈ average of [2.0, 0.0, ...] — not exactly 2, but close.

Sliced Wasserstein is an approximation — it's a lower bound on W₁ and converges
to the true W as L → ∞. For d >> 1 it's often the only practical option.

---

## 6. Summary

| Variant | Formula | Complexity | When to use |
|---|---|---|---|
| W_p (discrete, LP) | min_{P∈U} ⟨C,P⟩ | O((nm)³) | small n,m, exact needed |
| W_p (discrete, Sinkhorn) | regularized LP | O(nm·T) | medium n,m, differentiable |
| W_p (1D, quantile) | ∫|F⁻¹_μ−F⁻¹_ν|^p | O(n log n) | 1D distributions |
| W₂ (Gaussians) | closed form via Bures | O(d³) | Gaussian distributions |
| Sliced W_p | average 1D W_p | O(L·n log n·d) | high-dimensional, approximate |

**What's next:** Phase 5 uses W₂ to define the Fréchet mean of distributions
(the Wasserstein barycenter) — the "average" of a set of distributions in Wasserstein
space. This is the mathematical backbone of the flow matching interpolation in Phase 6.
