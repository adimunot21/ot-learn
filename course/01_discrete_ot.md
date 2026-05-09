# Chapter 1 — Discrete Optimal Transport as a Linear Program

*New to probability distributions or linear programs? Read `course/prerequisites.md` first — it covers exactly what you need. Keep `course/glossary.md` open for quick lookups.*

## The Problem and Why It Matters

Imagine three bakeries and three cafes. Each bakery has a certain supply of bread.
Each cafe has a certain demand. Moving bread costs money — proportional to distance.
How do you ship the bread to minimize total transport cost?

This is **Optimal Transport** in its simplest form. The word "optimal" means cheapest.
The word "transport" means we're moving mass (bread, probability, anything) from one
distribution to another.

Why does this matter beyond bread?

- **Machine learning**: comparing two probability distributions (e.g., real images vs
  generated images) in a way that respects geometry. KL divergence ignores geometry;
  OT does not.
- **Domain adaptation**: aligning a source dataset to a target dataset
- **Generative models**: the Wasserstein distance used in WGANs and flow matching is
  an OT cost (we'll build those in Phases 4 and 6)
- **Shape analysis, biology, economics, fluid dynamics** — anywhere two distributions
  need to be compared or transformed

The discrete case (finite points, weights) is the entry point. Everything continuous
we build later is a limit or extension of this.

---

## The Setup

We have two discrete distributions:

```
Source:  n points x₁, …, xₙ  with weights a ∈ ℝⁿ,  aᵢ ≥ 0,  Σᵢ aᵢ = 1
Target:  m points y₁, …, yₘ  with weights b ∈ ℝᵐ,  bⱼ ≥ 0,  Σⱼ bⱼ = 1
```

And a **cost matrix** C ∈ ℝⁿˣᵐ, where Cᵢⱼ = cost of moving one unit of mass from
xᵢ to yⱼ. Usually this is a distance: Cᵢⱼ = ‖xᵢ − yⱼ‖ or ‖xᵢ − yⱼ‖².

---

## Attempt 1: The Monge Map (1781)

Gaspard Monge's original formulation: find a **function** T: {x₁,…,xₙ} → {y₁,…,yₘ}
that assigns each source point to exactly one target point, subject to:

1. Mass is conserved: for each j, the total weight of sources mapped to yⱼ equals bⱼ
2. Total cost is minimized: min_T Σᵢ aᵢ · C(xᵢ, T(xᵢ))

The problem: **T can't split mass**. If a single source point holds all the mass,
it can only go to one target. But the target might demand that mass be split.

```
Example:
  a = [1.0]         (one source holds everything)
  b = [0.5, 0.5]    (two targets each want half)
  C = [[1, 2]]

No map T satisfies this. T(x₁) = y₁ leaves y₂ empty. T(x₁) = y₂ leaves y₁ empty.
Monge is stuck.
```

This is not an edge case. It happens whenever source and target distributions have
different "shapes" — and that's almost always true in practice.

---

## Kantorovich's Fix: The Transport Plan (1942)

Leonid Kantorovich's insight — instead of a rigid map, describe transport as a
**joint distribution** P over (source, target) pairs.

P is an n×m matrix. **Pᵢⱼ = amount of mass shipped from source i to target j.**

```
Text diagram — what P represents:

  source 1 ──── P₁₁ ────► target 1
             └── P₁₂ ────► target 2
  source 2 ──── P₂₁ ────► target 1
             └── P₂₂ ────► target 2
```

Three constraints:

```
(1) Pᵢⱼ ≥ 0                     can't move negative mass
(2) Σⱼ Pᵢⱼ = aᵢ  for all i      source i must ship out exactly aᵢ
(3) Σᵢ Pᵢⱼ = bⱼ  for all j      target j must receive exactly bⱼ
```

Constraints (2) and (3) are the **marginal constraints** — they say that if you
sum P over columns you get a, and if you sum over rows you get b.

Objective: minimize total cost

```
min_{P}  Σᵢ Σⱼ Cᵢⱼ Pᵢⱼ  =  min_{P}  ⟨C, P⟩
```

where ⟨C, P⟩ = Σᵢⱼ Cᵢⱼ Pᵢⱼ is the elementwise dot product.

**This is always feasible.** The "independent coupling" P = abᵀ (outer product)
always satisfies the marginals — it just ignores the cost.
Since the feasible set is non-empty and bounded, the minimum always exists.

**Monge as a special case.** A Monge map corresponds to P where each row has at most
one non-zero entry (each source sends its mass to exactly one target). Kantorovich
allows every entry to be non-zero — it's strictly more expressive.

---

## This Is a Linear Program

**Variables:** the nm entries of P, which we flatten into a vector p ∈ ℝⁿᵐ.

Row-major: `p[i*m + j] = P[i,j]`

**Objective:** c = C.flatten(), minimize cᵀp.

**Equality constraints** (the marginal constraints):

```
Source marginal i:   Σⱼ P[i,j] = aᵢ
→ In p: indices {i*m, i*m+1, …, i*m+(m-1)} sum to aᵢ
→ These are contiguous entries — one row of P.

Target marginal j:   Σᵢ P[i,j] = bⱼ
→ In p: indices {j, m+j, 2m+j, …, (n-1)*m+j} sum to bⱼ
→ These are strided entries — one column of P.
```

Stack these into matrix form:

```
A_eq ∈ ℝ^{(n+m) × nm}    (n source constraints + m target constraints)
b_eq = [a; b] ∈ ℝⁿ⁺ᵐ

min  cᵀp
s.t. A_eq p = b_eq
     p ≥ 0
```

Only n+m−1 constraints are independent (one is redundant because Σaᵢ = Σbⱼ = 1
implies the last constraint follows from all others). The LP solver handles this
automatically.

---

## Worked Example: 2×2

```
a = [0.6, 0.4]
b = [0.5, 0.5]

C = [[1, 3],    ← C[0,0]=1 (cheap), C[0,1]=3 (expensive)
     [2, 1]]    ← C[1,0]=2 (medium), C[1,1]=1 (cheap)
```

The LP has 4 variables: P₁₁, P₁₂, P₂₁, P₂₂.

Constraints written out:
```
P₁₁ + P₁₂        = 0.6    row 1 must sum to a₁
       P₂₁ + P₂₂ = 0.4    row 2 must sum to a₂
P₁₁       + P₂₁  = 0.5    col 1 must sum to b₁
       P₁₂ + P₂₂ = 0.5    col 2 must sum to b₂
Pᵢⱼ ≥ 0
```

**Solving by reasoning about the costs:**
The cheap paths are C₁₁=1 and C₂₂=1. We want to route as much mass as possible
through these. The expensive path is C₁₂=3 — minimize its use.

```
Step 1: maximize P₁₁ (cost 1):
  P₁₁ ≤ min(a₁, b₁) = min(0.6, 0.5) = 0.5
  Set P₁₁ = 0.5.

Step 2: remaining mass from source 1:  a₁ - P₁₁ = 0.6 - 0.5 = 0.1
  This 0.1 must go somewhere. b₁ is now full (0.5 used). Only target 2 is available.
  P₁₂ = 0.1.

Step 3: remaining demand at target 1:  b₁ - P₁₁ = 0.5 - 0.5 = 0.0
  P₂₁ = 0.0.

Step 4: source 2 has 0.4 and target 2 has 0.5 - 0.1 = 0.4 remaining.
  P₂₂ = 0.4.
```

**Solution:**
```
P* = [[0.5, 0.1],
      [0.0, 0.4]]

Row sums: [0.6, 0.4] = a  ✓
Col sums: [0.5, 0.5] = b  ✓

Cost = 0.5·1 + 0.1·3 + 0.0·2 + 0.4·1
     = 0.5  + 0.3  + 0.0  + 0.4
     = 1.2
```

Compare to the naive plan P = abᵀ:
```
P_naive = [[0.3, 0.3],   (just multiply: 0.6·0.5, 0.6·0.5, 0.4·0.5, 0.4·0.5)
           [0.2, 0.2]]

Cost_naive = 0.3·1 + 0.3·3 + 0.2·2 + 0.2·1 = 0.3 + 0.9 + 0.4 + 0.2 = 1.8
```

The optimal plan costs **1.2** vs the naive plan's **1.8** — a 33% saving by routing
mass through the cheap paths.

---

## The 3×3 Example

```
a = [0.5, 0.3, 0.2]      (bakery supply fractions)
b = [0.4, 0.4, 0.2]      (cafe demand fractions)

C = [[1, 2, 4],           bakery 1 is close to cafe 1, far from cafe 3
     [3, 1, 2],           bakery 2 is close to cafe 2
     [4, 3, 1]]           bakery 3 is close to cafe 3
```

Running our LP solver produces:
```
P* = [[0.4, 0.1, 0.0],
      [0.0, 0.3, 0.0],
      [0.0, 0.0, 0.2]]

Cost = 1.1
```

This matches the greedy trace from the derivation notes. The plan is nearly diagonal —
each bakery ships to its nearest cafe, with the overflow from bakery 1 going to cafe 2
(cost 2, the next cheapest option).

---

## The Code, Line by Line

### Building the constraint matrix

```python
def build_constraint_matrix(n: int, m: int) -> np.ndarray:
    nm = n * m
    A_eq = np.zeros((n + m, nm))

    # Source marginals: each row i of P is a contiguous block in p
    for i in range(n):
        A_eq[i, i * m : i * m + m] = 1.0

    # Target marginals: each col j of P is a strided slice of p
    for j in range(m):
        A_eq[n + j, j::m] = 1.0

    return A_eq  # shape: (n+m, n*m)
```

The `j::m` stride is the key insight: when p is row-major and P has m columns,
column j appears at positions j, m+j, 2m+j, ... — exactly a stride of m.

### The solver

```python
def solve_ot(cost_matrix, source_weights, target_weights):
    n, m = cost_matrix.shape
    c    = cost_matrix.flatten()              # (n*m,)  objective
    A_eq = build_constraint_matrix(n, m)     # (n+m, n*m)  constraints
    b_eq = np.concatenate([source_weights,   # (n+m,)  RHS
                           target_weights])
    bounds = [(0.0, None)] * (n * m)         # Pᵢⱼ ≥ 0

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    transport_plan = result.x.reshape(n, m)  # (n,m)  un-flatten P
    return transport_plan, float(result.fun)
```

We use HiGHS (the `method="highs"` flag), which is the fastest LP solver bundled
with scipy. For small n, m this solves in milliseconds.

---

## Shape Traces

| Variable | Shape | Meaning |
|---|---|---|
| `C` | `(n, m)` | cost matrix |
| `a` | `(n,)` | source weights |
| `b` | `(m,)` | target weights |
| `p = C.flatten()` | `(n*m,)` | LP objective vector |
| `A_eq` | `(n+m, n*m)` | marginal constraint matrix |
| `b_eq` | `(n+m,)` | marginal constraint RHS |
| `result.x` | `(n*m,)` | optimal transport plan, flattened |
| `transport_plan` | `(n, m)` | optimal transport plan, reshaped |

---

## Summary Table

| Concept | Definition |
|---|---|
| Transport plan P | n×m matrix; Pᵢⱼ = mass from source i to target j |
| Marginal constraints | Row sums = a; column sums = b |
| Transport polytope U(a,b) | Convex set of all feasible P |
| OT cost | ⟨C, P⟩ = Σᵢⱼ Cᵢⱼ Pᵢⱼ |
| Monge map | Rigid map; special case where each row of P has one nonzero |
| Kantorovich relaxation | Allows mass splitting; always feasible; strictly more general |
| LP formulation | nm variables, n+m−1 independent equality constraints |
| Product measure a⊗b | Baseline plan; marginals correct but ignores cost |

---

## What's Next

We solved the primal problem: find the cheapest P.

But every LP has a **dual problem** — and the dual of OT has a beautiful meaning.
The dual variables are "prices" assigned to each source and target location.
At the optimum, these prices satisfy a remarkable complementarity condition:
you only use a shipping route if it's cost-competitive given the prices.

This is Phase 2: Kantorovich Duality.
