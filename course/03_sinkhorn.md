# Chapter 3 — The Sinkhorn Algorithm

## The Problem and Why It Matters

The LP solver from Phase 1 is exact but doesn't scale.
A 1000×1000 problem (comparing two batches of 1000 samples each) has 10⁶ variables.
The LP takes minutes. Backpropagation through it is impossible.

The Sinkhorn algorithm solves a slightly modified OT problem in O(nm) per iteration
using only matrix-vector products — GPU-friendly, differentiable, and fast.
It powers virtually every modern use of OT in machine learning:
Wasserstein distances, Wasserstein barycenters, and the OT coupling in flow matching.

The modification: add an entropy term to the objective.

---

## The Regularized Problem

Instead of:

```
min_{P ∈ U(a,b)}  <C, P>
```

we solve:

```
min_{P ∈ U(a,b)}  <C, P>  +  ε · KL(P ‖ a⊗b)
```

where ε > 0 is a tunable regularization strength and:

```
KL(P ‖ a⊗b) = Σᵢⱼ Pᵢⱼ · log(Pᵢⱼ / (aᵢ bⱼ))
```

This is the KL divergence from P to the "reference plan" a⊗b (the independent coupling).

**Why KL divergence?** Because it enforces P > 0 everywhere (log 0 = −∞ acts as a
barrier), makes the problem strictly convex (unique solution), and — as we'll see —
its dual leads to a beautiful multiplicative update rule.

**The entropy view.** KL(P‖a⊗b) = −H(P) + constant, so:

```
min <C, P> + ε · KL(P‖a⊗b)  ≡  min <C, P> − ε · H(P)
```

The entropy term `−ε · H(P)` penalizes sparse plans. Without it, the optimal P has
at most n+m−1 non-zeros (a vertex of the polytope). With it, every entry is positive.

```
Text diagram — effect of ε on the plan:

  ε → ∞    P → a⊗b           (mass spreads uniformly, ignores cost)
  ε = 1.0  P ≈ blurred OT    (smooth plan, easy to converge)
  ε = 0.1  P ≈ sharp OT      (close to exact, 91 iters on 2×2)
  ε → 0    P → exact OT      (but convergence needs ∝ 1/ε iterations)
```

---

## Deriving the Sinkhorn Iterations

The regularized problem has a dual:

```
max_{u,v}  a^T u + b^T v  −  ε Σᵢⱼ aᵢbⱼ exp((uᵢ+vⱼ−Cᵢⱼ)/ε)  +  ε
```

Setting the gradient to zero (the optimality condition) recovers the primal solution:

```
Pᵢⱼ* = aᵢ bⱼ exp((uᵢ + vⱼ − Cᵢⱼ) / ε)                     ...(★)
```

This is the key formula: **the optimal regularized plan is exponential in the dual
potentials**. Now we use the marginal constraints to find u and v.

**Define the Gibbs kernel** K ∈ ℝⁿˣᵐ (fixed for the entire run):

```
Kᵢⱼ = exp(−Cᵢⱼ / ε)
```

**Define scaling vectors** (related to dual potentials by f = exp(u/ε), g = exp(v/ε)):

```
fᵢ = exp(uᵢ/ε)    →   f ∈ ℝⁿ₊
gⱼ = exp(vⱼ/ε)    →   g ∈ ℝᵐ₊
```

The optimal plan (★) can be written as:

```
P* = diag(f) · K · diag(g)         ← "Sinkhorn form"

Pᵢⱼ = fᵢ · Kᵢⱼ · gⱼ
```

The plan is just the kernel K with its rows rescaled by f and columns by g.

**Source marginal constraint** (row i of P* must sum to aᵢ):

```
Σⱼ fᵢ Kᵢⱼ gⱼ = aᵢ
fᵢ · (K g)ᵢ = aᵢ
fᵢ = aᵢ / (K g)ᵢ
```

**Target marginal constraint** (col j of P* must sum to bⱼ):

```
Σᵢ fᵢ Kᵢⱼ gⱼ = bⱼ
gⱼ · (Kᵀ f)ⱼ = bⱼ
gⱼ = bⱼ / (Kᵀ f)ⱼ
```

These two updates are the **Sinkhorn iterations**:

```
f ← a / (K  @ g)       (n,) ← (n,) / (n,m) @ (m,)
g ← b / (K.T @ f)      (m,) ← (m,) / (m,n) @ (n,)
```

Each enforces one marginal exactly while potentially disturbing the other.
Alternate until convergence. That's it.

---

## The Full Algorithm

```
Input:  C (n,m), a (n,), b (m,), ε, max_iter
Output: P ≈ optimal regularized transport plan (n,m)

1. K = exp(-C / ε)             (n,m)  precompute, fixed throughout
2. g = ones(m)                 (m,)   any positive initialization
3. repeat:
      f = a / (K  @ g)         (n,)   enforce source marginal
      g = b / (K.T @ f)        (m,)   enforce target marginal
4. P = diag(f) @ K @ diag(g)   (n,m)  = f[:,None] * K * g[None,:]
5. return P, sum(C * P)
```

**Complexity:** Step 3 costs O(nm) per round (two matrix-vector products).
Total: O(nm · T) where T is iterations. Compare to O((nm)³) for the LP.

---

## Worked Example: 2×2, ε = 1.0

```
a = [0.6, 0.4],  b = [0.5, 0.5],  ε = 1.0

C = [[1, 3],
     [2, 1]]

K = exp(-C/1.0) = [[e⁻¹, e⁻³],   ≈  [[0.3679, 0.0498],
                    [e⁻², e⁻¹]]        [0.1353, 0.3679]]
```

**Iteration 1** (g initialised to [0.5, 0.5]):

```
K @ g = [0.3679·0.5 + 0.0498·0.5,  0.1353·0.5 + 0.3679·0.5]
      = [0.2089,  0.2516]
f = a / (K @ g) = [0.6/0.2089, 0.4/0.2516] = [2.873, 1.590]

K.T @ f = [0.3679·2.873 + 0.1353·1.590,  0.0498·2.873 + 0.3679·1.590]
        = [1.057 + 0.215,   0.143 + 0.585]
        = [1.272, 0.728]
g = b / (K.T @ f) = [0.5/1.272, 0.5/0.728] = [0.393, 0.687]
```

**After many iterations**, f and g converge. Reconstructed plan at ε=1.0:

```
P_ε=1 ≈ [[0.417, 0.183],    (compare to exact [[0.5, 0.1],
          [0.083, 0.317]]                        [0.0, 0.4]])
```

The plan is blurred — P[1,0] = 0.083 instead of 0. Cost ≈ 1.354 > 1.200.

At ε=0.1, the plan sharpens to essentially the exact LP solution (cost=1.200).

---

## The Numerical Stability Problem

For small ε (say ε=0.01), Kᵢⱼ = exp(−Cᵢⱼ/ε) can underflow to 0 for large Cᵢⱼ.

```
C[1,0] = 3.0,  ε = 0.001:   exp(-3000) = 0.0   (underflow)
```

Then `K @ g` is zero, `f = a / 0 = inf`, and the algorithm produces NaN.

Observed in practice:
```
ε=0.010:  cost=1.200000  marginal_err=8.9e-10  iters=535   ← fine
ε=0.001:  cost=nan       marginal_err=nan       iters=2000  ← broken
```

---

## The Log-Domain Fix

Work entirely in log-space. Define log-potentials:

```
u = ε · log(f)    (n,)
v = ε · log(g)    (m,)
```

The Sinkhorn updates in log-space:

```
u ← ε · log(a)  −  ε · logsumexp(−C/ε + v[None,:]/ε, axis=1)
v ← ε · log(b)  −  ε · logsumexp(−C/ε + u[:,None]/ε, axis=0)
```

`logsumexp(x)` = log(Σᵢ exp(xᵢ)) computed using the max-subtraction trick,
so it never overflows or underflows.

Recover the plan only at the very end:

```
log P = u[:,None]/ε + v[None,:]/ε − C/ε     (n,m)
P = exp(log P)
```

**This is numerically identical to vanilla Sinkhorn** — just computed in a way that
never produces 0 or inf intermediates.

### Code side-by-side

```python
# Vanilla (breaks for small ε)
K = np.exp(-C / epsilon)
f = a / (K @ g)
g = b / (K.T @ f)

# Log-domain (stable for any ε)
M = -C / epsilon                                  # (n,m) log-kernel
u = epsilon * (log_a - logsumexp(M + v/epsilon, axis=1))
v = epsilon * (log_b - logsumexp(M + u/epsilon, axis=0))
```

The logsumexp in the log-domain update is exactly the soft version of the c-transform
from Phase 2:

```
Hard c-transform:  v_j = min_i (C_ij - u_i)           (Phase 2 dual)
Soft c-transform:  v_j ≈ -ε · logsumexp_i(-C_ij/ε + u_i/ε)   (Sinkhorn dual)
```

As ε→0, logsumexp → max, and max(−C_ij/ε + u_i/ε) = −(1/ε) · min_i(C_ij − u_i).
The Sinkhorn dual **converges to the exact dual** as ε→0. Duality connects everything.

---

## Convergence Behaviour

```
[Convergence plot: notebooks/sinkhorn_convergence.png]

ε = 2.0:  converges in ~10 iterations   (fast, blurry plan)
ε = 0.5:  converges in ~30 iterations
ε = 0.1:  converges in ~150 iterations
ε = 0.01: converges in ~2000 iterations  (slow, sharp plan)
```

Rule of thumb: number of iterations scales as ~1/ε.
For most ML use cases, ε=0.1 is a good default — recovers near-exact OT, converges in
tens to hundreds of iterations.

---

## The Code, Line by Line

### Log-domain Sinkhorn

```python
def log_sinkhorn(cost_matrix, source_weights, target_weights, epsilon, max_iter, tol):
    n, m = cost_matrix.shape
    log_a = np.log(source_weights)        # (n,)
    log_b = np.log(target_weights)        # (m,)
    M = -cost_matrix / epsilon            # (n,m) log-kernel, fixed

    u = np.zeros(n)                       # (n,) log-potentials, initialise to 0
    v = np.zeros(m)                       # (m,)

    for _ in range(max_iter):
        # Source update: u_i = ε·log(a_i) - ε·logsumexp_j(M_ij + v_j/ε)
        # M + v[None,:]/ε  shape: (n,m)
        # logsumexp over axis=1 → (n,)
        u = epsilon * (log_a - logsumexp(M + v[None,:] / epsilon, axis=1))

        # Target update: v_j = ε·log(b_j) - ε·logsumexp_i(M_ij + u_i/ε)
        # M + u[:,None]/ε  shape: (n,m)
        # logsumexp over axis=0 → (m,)
        v = epsilon * (log_b - logsumexp(M + u[:,None] / epsilon, axis=0))

    # Recover plan in log-space, only exponentiate at the end
    log_P = u[:,None]/epsilon + v[None,:]/epsilon + M    # (n,m)
    transport_plan = np.exp(log_P)                        # (n,m)
    return transport_plan
```

### Shape traces

| Expression | Shape | Operation |
|---|---|---|
| `M = -C/ε` | `(n,m)` | fixed log-kernel |
| `v[None,:]/ε` | `(1,m)` | broadcast to `(n,m)` when added to M |
| `M + v[None,:]/ε` | `(n,m)` | input to logsumexp |
| `logsumexp(..., axis=1)` | `(n,)` | sum over targets |
| `u = ε*(log_a - ...)` | `(n,)` | new source potentials |
| `u[:,None]/ε` | `(n,1)` | broadcast to `(n,m)` |
| `log_P` | `(n,m)` | log of transport plan |
| `transport_plan` | `(n,m)` | final plan |

---

## Summary Table

| Concept | Definition |
|---|---|
| Regularized OT | `min <C,P> + ε·KL(P‖a⊗b)` s.t. marginals |
| Gibbs kernel K | `K_ij = exp(-C_ij/ε)` — fixed for entire run |
| Sinkhorn form | `P* = diag(f) K diag(g)` |
| f, g updates | `f ← a/(Kg)`, `g ← b/(K^T f)` |
| Log-domain | `u = ε log f`, `v = ε log g`; uses logsumexp |
| Soft c-transform | `v_j = -ε logsumexp_i(-C_ij/ε + u_i/ε)` |
| ε large | Fast convergence, blurred plan, higher cost |
| ε small | Slow convergence, sharp plan, approaches exact OT |

---

## What's Next

We now have an efficient way to compute near-optimal transport plans.
But often we don't need the plan — we just need the **cost** at optimum.

The minimum OT cost between two distributions is the **Wasserstein distance**.
In Phase 4 we study its properties, its 1D closed form (no solver needed),
and the sliced approximation that makes it practical in high dimensions.
