# Chapter 2 — Kantorovich Duality

## The Problem and Why It Matters

In Phase 1 we solved the **primal** OT problem: find the cheapest transport plan P.

Every linear program has a twin — the **dual problem** — that optimizes over a
completely different set of variables. For most LPs the dual is a mathematical
curiosity. For OT, the dual is a fundamental object with a concrete meaning,
and it unlocks the Sinkhorn algorithm (Phase 3), the Wasserstein distance (Phase 4),
and ultimately flow matching (Phase 6).

The dual of OT answers this question: **what is the maximum you could charge for
shipping, if you're not allowed to overcharge anyone?**

---

## Quick LP Duality Refresher

Given a primal LP:

```
min  c^T p
s.t. A p  = b
     p   >= 0
```

The dual is:

```
max  b^T λ
s.t. A^T λ <= c
```

where λ is one variable per equality constraint.

**Weak duality** (always true, follows from algebra):
for any feasible primal p and feasible dual λ,

```
b^T λ = (Ap)^T λ = p^T (A^T λ) ≤ p^T c
```

So the dual objective is always a lower bound on the primal cost.

**Strong duality**: at the optimum, they're equal — no gap.
This holds for OT because the primal is always feasible and bounded.

---

## Deriving the OT Dual

The primal OT LP has:
- `nm` variables (the transport plan, flattened)
- `n + m` equality constraints (marginals)
- constraint matrix A_eq of shape `(n+m, nm)`

Dual variables: one per primal constraint, so `n + m` total.
Split them: `u ∈ ℝⁿ` for source constraints, `v ∈ ℝᵐ` for target constraints.

The dual constraint says `A_eq^T [u; v] ≤ c`. What does this say for variable P_ij?

Column `(i*m + j)` of A_eq has exactly two 1s: at row i (source i) and row n+j (target j).
So the dual constraint for P_ij is:

```
u_i + v_j  <=  C_ij
```

The dual objective is `b_eq^T [u; v] = a^T u + b^T v`.

**The complete OT dual:**

```
max_{u ∈ ℝⁿ, v ∈ ℝᵐ}   a^T u + b^T v
s.t.                     u_i + v_j <= C_ij   for all i, j
```

A dramatic reduction: from `nm` variables to `n + m`.
For a 100×100 problem: 10,000 primal variables → 200 dual variables.

---

## Economic Interpretation

You run a shipping company. You set:
- `u_i` = fee to pick up one unit at source i
- `v_j` = fee to deliver one unit to target j

A customer shipping from i to j pays `u_i + v_j`.

The dual constraint `u_i + v_j ≤ C_ij` says: **you can't charge more than the
actual transport cost**. If you did, the customer would self-transport.

The dual objective `a^T u + b^T v` is your total revenue.

The dual asks: **maximize revenue subject to competitive pricing**.

Strong duality says: at the optimum, the maximum competitive revenue equals
the minimum transport cost. The invisible hand of LP theory.

---

## Complementary Slackness

This is the most important structural result of duality. At any primal-dual optimum:

```
P_ij · (C_ij - u_i - v_j) = 0     for all i, j
```

One of the two factors must be zero. Two cases:

```
Case 1: P_ij > 0  (mass flows along route i→j)
    => C_ij - u_i - v_j = 0
    => u_i + v_j = C_ij
    The route "breaks even" — price exactly equals cost.

Case 2: u_i + v_j < C_ij  (route is "unprofitable" at current prices)
    => P_ij = 0
    No mass flows on overpriced routes.
```

In plain English: **mass only flows where the price equals the cost**.

```
Text diagram:

  source i ───[P_ij > 0]───► target j    requires   u_i + v_j = C_ij  (tight)
  source i ───[P_ij = 0]───► target j    allowed    u_i + v_j < C_ij  (slack)
```

This will be the key to understanding Sinkhorn: the algorithm enforces a *softened*
version of this, where instead of exactly-zero mass on slack routes, we allow
exponentially small mass.

---

## The c-Transform

Given source potentials `u ∈ ℝⁿ`, the **c-transform** gives the tightest feasible
target potentials:

```
v_j = (u^c)_j = min_i (C_ij - u_i)
```

This is the maximum `v_j` can be while satisfying `u_i + v_j ≤ C_ij` for all i.

```python
def c_transform(u, C):
    # C - u[:, None] broadcasts to (n, m)
    # min over axis 0 takes the cheapest source for each target
    return (C - u[:, None]).min(axis=0)    # (n,m) → (m,)
```

Given any u, applying the c-transform gives the best v. Then apply it again
(with roles of u and v swapped) to get the best u given v. This is an alternating
minimization that converges to the dual optimum.

In Phase 3, the c-transform becomes a **log-sum-exp** (a soft minimum) — and that's
exactly the Sinkhorn update.

---

## Worked Example: 2×2

```
a = [0.6, 0.4]
b = [0.5, 0.5]

C = [[1, 3],
     [2, 1]]

Primal optimal: P* = [[0.5, 0.1], [0.0, 0.4]],  cost = 1.2
```

**Finding u and v by complementary slackness.**

Active routes (P_ij > 0): (0,0), (0,1), (1,1). Three equations:

```
u_0 + v_0 = C_00 = 1    ...(A)
u_0 + v_1 = C_01 = 3    ...(B)
u_1 + v_1 = C_11 = 1    ...(C)
```

Four unknowns, three equations → one free parameter (expected: dual solution
is unique only up to a constant shift — adding δ to all u and subtracting δ
from all v changes nothing).

Fix u_0 = 0:
```
(A): v_0 = 1
(B): v_1 = 3
(C): u_1 = 1 - 3 = -2
```

**Solution: u = [0, -2],  v = [1, 3]**

Verify all dual constraints `u_i + v_j <= C_ij`:
```
(0,0): 0 + 1 = 1 <= 1  ✓  tight (route used)
(0,1): 0 + 3 = 3 <= 3  ✓  tight (route used)
(1,0): -2 + 1 = -1 <= 2  ✓  slack (route unused, P_10 = 0)
(1,1): -2 + 3 = 1 <= 1  ✓  tight (route used)
```

Verify strong duality:
```
Dual objective = a^T u + b^T v
              = 0.6·(0) + 0.4·(-2)  +  0.5·(1) + 0.5·(3)
              = 0 - 0.8 + 0.5 + 1.5
              = 1.2  ✓  equals primal cost
```

---

## What the Code Does, Line by Line

### Building and solving the dual LP

```python
def solve_dual_ot(cost_matrix, source_weights, target_weights):
    n, m = cost_matrix.shape

    # Negate objective: scipy minimises, we want to maximise a^T u + b^T v
    c_dual = np.concatenate([-source_weights, -target_weights])  # (n+m,)

    # One inequality row per (i,j) pair: u_i + v_j <= C_ij
    A_ub = np.zeros((n * m, n + m))     # (n*m, n+m)
    for i in range(n):
        for j in range(m):
            row = i * m + j
            A_ub[row, i]     = 1.0     # coefficient of u_i
            A_ub[row, n + j] = 1.0     # coefficient of v_j
    b_ub = cost_matrix.flatten()       # (n*m,) — RHS of inequalities

    # Dual variables are unbounded (can be negative — source 2 has u=-2 above)
    bounds = [(None, None)] * (n + m)

    result = linprog(c_dual, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    u = result.x[:n]             # (n,) source potentials
    v = result.x[n:]             # (m,) target potentials
    dual_value = -result.fun     # negate back
    return u, v, dual_value
```

### Checking complementary slackness

```python
def check_complementary_slackness(P, u, v, C, tol=1e-6):
    slack  = C - u[:, None] - v[None, :]    # (n,m)  how much each route is overpriced
    active = P > tol                         # (n,m)  which routes carry mass
    # Violation: route is active AND has positive slack (not tight)
    violations = active & (slack > tol)
    return not violations.any()
```

---

## Shape Traces

| Variable | Shape | Meaning |
|---|---|---|
| `u` | `(n,)` | source dual potentials |
| `v` | `(m,)` | target dual potentials |
| `c_dual` | `(n+m,)` | negated dual objective coefficients |
| `A_ub` | `(n*m, n+m)` | dual inequality constraint matrix |
| `b_ub` | `(n*m,)` | RHS = C.flatten() |
| `slack = C - u[:,None] - v[None,:]` | `(n, m)` | how much each route is overpriced |

---

## Summary Table

| Concept | Definition |
|---|---|
| Dual variables (u, v) | Prices at sources and targets |
| Dual constraint | `u_i + v_j <= C_ij` — can't overcharge |
| Dual objective | `a^T u + b^T v` — total revenue |
| Weak duality | Dual obj ≤ primal cost at every feasible point |
| Strong duality | At optimum: dual obj = primal cost |
| Duality gap | `primal_cost - dual_value` — should be ~0 at optimum |
| Complementary slackness | `P_ij > 0 ⟹ u_i + v_j = C_ij` |
| c-transform | `v_j = min_i(C_ij - u_i)` — tightest feasible v given u |
| Degree of freedom | Dual solution defined up to a global constant shift |

---

## What's Next

We've been solving OT exactly — but exact LP solvers don't scale.
A 1000×1000 problem has 10⁶ variables and takes seconds on a laptop.
Real ML applications deal with millions of samples.

The fix: **add entropy to the objective**. Entropy regularization makes the problem
strictly convex, breaks it into a beautifully simple iterative algorithm (Sinkhorn),
and makes it GPU-friendly. The cost: we solve a slightly different problem.
But in Phase 3 we'll see exactly how different, and how to control the error.
