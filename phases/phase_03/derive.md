# Phase 3 Derivation — The Sinkhorn Algorithm

## Why Not Just Use the LP?

The LP solver from Phases 1–2 is exact but scales badly.
Variables grow as n·m. Constraints grow as n+m. Solver complexity is roughly O((nm)³).

| Problem size | Variables | LP solve time (rough) |
|---|---|---|
| 10×10 | 100 | < 1ms |
| 100×100 | 10,000 | ~100ms |
| 1000×1000 | 1,000,000 | hours |
| 10k×10k | 100,000,000 | not feasible |

In machine learning we compare batches of 1000+ samples routinely.
We need an algorithm that is:
1. Fast (ideally O(nm) per iteration)
2. Parallelizable (GPU-friendly matrix ops)
3. Differentiable (so we can backprop through OT costs)

The answer: add entropy to the objective. This is Sinkhorn.

---

## 1. Entropy-Regularized OT

Instead of solving:

    min_{P ∈ U(a,b)}  ⟨C, P⟩

we solve:

    min_{P ∈ U(a,b)}  ⟨C, P⟩ + ε · KL(P ‖ a⊗b)

where:
- ε > 0 is the **regularization strength**
- KL(P ‖ Q) = Σᵢⱼ Pᵢⱼ log(Pᵢⱼ / Qᵢⱼ) is the KL divergence
- a⊗b is the product measure (the "reference" plan): (a⊗b)ᵢⱼ = aᵢ bⱼ

Unpacking the KL term:

    KL(P ‖ a⊗b) = Σᵢⱼ Pᵢⱼ log(Pᵢⱼ / (aᵢ bⱼ))
                = Σᵢⱼ Pᵢⱼ log Pᵢⱼ  −  Σᵢⱼ Pᵢⱼ log aᵢ  −  Σᵢⱼ Pᵢⱼ log bⱼ
                = −H(P) − aᵀlog a − bᵀlog b

where H(P) = −Σᵢⱼ Pᵢⱼ log Pᵢⱼ is the entropy of P.

Since aᵀlog a + bᵀlog b is a constant w.r.t. P, minimizing ⟨C,P⟩ + ε·KL(P‖a⊗b)
is equivalent to:

    min_{P ∈ U(a,b)}  ⟨C, P⟩ − ε · H(P)

The entropy term **penalizes sparse plans**. Without it, the optimal P has at most
n+m−1 nonzero entries (it's a vertex of the transport polytope).
With it, all entries are positive — mass is "smeared" across all routes.

As ε → 0: the regularized solution approaches the exact OT solution.
As ε → ∞: the solution approaches the product measure a⊗b (full spreading).

---

## 2. The Regularized Dual

The regularized primal also has a dual. With entropy regularization, the dual becomes:

    max_{u, v}  aᵀu + bᵀv − ε Σᵢⱼ aᵢ bⱼ exp((uᵢ + vⱼ − Cᵢⱼ)/ε)  +  ε

where now there are no inequality constraints — the constraint uᵢ + vⱼ ≤ Cᵢⱼ
has been replaced by a smooth penalty term.

The unconstrained dual is differentiable and strictly concave.
Setting its gradient to zero recovers the primal solution via:

    Pᵢⱼ* = aᵢ bⱼ exp((uᵢ + vⱼ − Cᵢⱼ)/ε)         ...(*)

This is the key formula. The optimal regularized transport plan is an
**entrywise exponential** of the dual potentials, scaled by the marginals.

---

## 3. Deriving the Sinkhorn Iterations

From formula (*), the optimal P must satisfy the marginal constraints.

**Source marginal constraint** (row i sums to aᵢ):

    Σⱼ Pᵢⱼ* = Σⱼ aᵢ bⱼ exp((uᵢ + vⱼ − Cᵢⱼ)/ε) = aᵢ

    => exp(uᵢ/ε) · Σⱼ bⱼ exp((vⱼ − Cᵢⱼ)/ε) = 1

    => exp(uᵢ/ε) = 1 / Σⱼ bⱼ exp((vⱼ − Cᵢⱼ)/ε)

Define the **Gibbs kernel** K ∈ ℝⁿˣᵐ:

    Kᵢⱼ = exp(−Cᵢⱼ/ε)                                ...(K)

Then the constraint becomes:

    exp(uᵢ/ε) = 1 / Σⱼ (bⱼ exp(vⱼ/ε)) · Kᵢⱼ

Define rescaling vectors:

    fᵢ = exp(uᵢ/ε)  ∈ ℝ₊,   f = exp(u/ε) ∈ ℝⁿ₊
    gⱼ = exp(vⱼ/ε)  ∈ ℝ₊,   g = exp(v/ε) ∈ ℝᵐ₊

The optimal P factors as:

    P* = diag(f) · K · diag(g)                         ...(Sinkhorn form)

    Pᵢⱼ* = fᵢ · Kᵢⱼ · gⱼ = fᵢ · exp(−Cᵢⱼ/ε) · gⱼ

The marginal constraints in terms of f, g:

    Source: diag(f) K g = a   =>   f = a / (K g)   (elementwise division)
    Target: diag(g) Kᵀ f = b  =>   g = b / (Kᵀ f)  (elementwise division)

These two updates give the **Sinkhorn iterations**:

    f ← a / (K g)        [update source scaling]
    g ← b / (Kᵀ f)       [update target scaling]

That's the entire algorithm. Alternate between these two updates until convergence.

---

## 4. The Algorithm

```
Input:  C (n,m), a (n,), b (m,), ε > 0, max_iter
Output: P (n,m) ≈ optimal regularized transport plan

1. K = exp(-C / ε)                  (n,m) — compute once, fixed throughout
2. g = ones(m) / m                  (m,)  — initialize (any positive vector works)
3. for t = 1, 2, ..., max_iter:
   a. f = a / (K @ g)               (n,)  — source marginal constraint
   b. g = b / (K.T @ f)             (m,)  — target marginal constraint
4. P = diag(f) @ K @ diag(g)        (n,m) — reconstruct plan
5. return P, cost = sum(C * P)
```

Each iteration is two matrix-vector products: O(nm). Total: O(nm · T) where T is
iterations. This is linear in nm — a massive improvement over LP.

The matrix multiplications (step 3a, 3b) are parallel — they run on GPU natively.

---

## 5. Why Does It Converge?

Sinkhorn's algorithm is a coordinate ascent on the regularized dual.
Each update (f ← a/(Kg), g ← b/(Kᵀf)) exactly satisfies one marginal constraint
while potentially violating the other. After each full round, both marginals are
approximately satisfied, and the violation decreases geometrically.

More precisely: viewed in log space (u = ε log f, v = ε log g), the updates are:

    u ← ε log a − ε log(K exp(v/ε))
    v ← ε log b − ε log(Kᵀ exp(u/ε))

This is alternating projection onto two affine constraints in a
Bregman-divergence geometry. The convergence rate is linear, with contraction
factor related to the "spread" of K (its Hilbert projective metric diameter).

For our purposes: larger ε means faster convergence (smoother K, faster contraction).
Smaller ε means slower convergence (K more extreme, slower contraction).

---

## 6. Numerical Instability and the Log-Domain Fix

For small ε, Kᵢⱼ = exp(−Cᵢⱼ/ε) underflows to zero for large Cᵢⱼ.
Then f = a/(Kg) divides by zero. The algorithm breaks.

**Fix: work in log space throughout.**

Define log-potentials: u = ε log f, v = ε log g.
The updates in log space:

    u ← ε log a − ε · log_sum_exp_cols(v/ε − C/ε)
    v ← ε log b − ε · log_sum_exp_rows(u/ε − C/ε)

where log_sum_exp_cols(M) means log(Σⱼ exp(Mᵢⱼ)) applied per row
and log_sum_exp_rows(M) means log(Σᵢ exp(Mᵢⱼ)) applied per column.

These operations never underflow because numpy's logsumexp uses the
max-subtraction trick internally.

The log-domain Sinkhorn:

    u ← ε · (log a − logsumexp(-C/ε + v[None,:]/ε, axis=1))
    v ← ε · (log b − logsumexp(-C/ε + u[:,None]/ε, axis=0))

Recover P from u, v:
    log P = u[:,None]/ε + v[None,:]/ε − C/ε   (no exp until needed)

---

## 7. Worked Example — 2×2 with ε = 1.0

Using the same example from Phases 1 and 2:

```
a = [0.6, 0.4],  b = [0.5, 0.5],  ε = 1.0

C = [[1, 3],
     [2, 1]]

K = exp(-C/ε) = [[e⁻¹, e⁻³],   = [[0.3679, 0.0498],
                  [e⁻², e⁻¹]]      [0.1353, 0.3679]]
```

**Iteration 1:**

    g = [1, 1]   (initialization)

    f = a / (K g)
      K g = [0.3679·1 + 0.0498·1,   0.1353·1 + 0.3679·1]
           = [0.4177,  0.5032]
      f = [0.6/0.4177,  0.4/0.5032]
        = [1.4365,  0.7948]

    g = b / (Kᵀ f)
      Kᵀ f = [0.3679·1.4365 + 0.1353·0.7948,   0.0498·1.4365 + 0.3679·0.7948]
            = [0.5286 + 0.1075,   0.0716 + 0.2923]
            = [0.6361,  0.3639]
      g = [0.5/0.6361,  0.5/0.3639]
        = [0.7860,  1.3740]

**After 1 iteration:** f = [1.4365, 0.7948], g = [0.7860, 1.3740]

**Reconstruct P:**

    P = diag(f) · K · diag(g)

    P[0,0] = f[0]·K[0,0]·g[0] = 1.4365 · 0.3679 · 0.7860 = 0.4152
    P[0,1] = f[0]·K[0,1]·g[1] = 1.4365 · 0.0498 · 1.3740 = 0.0984
    P[1,0] = f[1]·K[1,0]·g[0] = 0.7948 · 0.1353 · 0.7860 = 0.0844
    P[1,1] = f[1]·K[1,1]·g[1] = 0.7948 · 0.3679 · 1.3740 = 0.4021

    P ≈ [[0.415, 0.098],   row sums ≈ [0.513, 0.486]  (not yet = [0.6, 0.4])
         [0.084, 0.402]]   col sums ≈ [0.499, 0.500]

After more iterations, marginals converge. At convergence (ε=1.0), the plan will
be "smeared" compared to the exact LP solution P*=[[0.5,0.1],[0,0.4]].
That's the tradeoff — regularization blurs the solution.

With ε→0, the Sinkhorn plan converges to the LP solution.

---

## 8. Key Takeaways

| Concept | What it means |
|---|---|
| Entropy regularization | Add −ε·H(P) to objective; penalizes sparse plans |
| Gibbs kernel K | Kᵢⱼ = exp(−Cᵢⱼ/ε); fixed for the whole run |
| Sinkhorn form | P* = diag(f) K diag(g); factored as row/col scalings |
| f, g updates | f ← a/(Kg), g ← b/(Kᵀf); each enforces one marginal |
| Log-domain | Work in u=ε log f, v=ε log g to avoid underflow |
| ε large | Fast convergence, plan is blurred (close to a⊗b) |
| ε small | Slow convergence, plan is sharp (close to exact OT) |
| Complexity | O(nm) per iteration vs O((nm)³) for LP |

**What's next:** Phase 4 uses Sinkhorn to compute Wasserstein distances — the OT cost
at the optimum. This is the number that WGANs optimize and flow matching relies on.
