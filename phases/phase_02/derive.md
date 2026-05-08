# Phase 2 Derivation — Kantorovich Duality

## Why Duality?

We solved the primal problem in Phase 1: find the cheapest transport plan P.
But every LP has a **dual problem** — a second optimization over a different set of
variables — and the dual of OT is where OT gets genuinely interesting.

The dual variables have a concrete economic meaning: they are **prices**.
The dual problem tells you: what is the maximum you could charge for shipping,
without incentivizing anyone to bypass the system?

And strong duality — the primal and dual optima are equal — gives us a tool we'll
use in every phase from here on.

---

## 1. LP Duality in 90 Seconds

Given the primal LP:

    min  cᵀp
    s.t. A_eq p = b_eq
         p ≥ 0

The dual LP is:

    max  b_eq^T λ
    s.t. A_eq^T λ ≤ c

where λ ∈ ℝⁿ⁺ᵐ is the vector of dual variables (one per equality constraint).

**Weak duality** (always true): any feasible dual λ gives a lower bound on the primal.
For any feasible p and λ:

    b_eq^T λ = (A_eq p)^T λ = p^T (A_eq^T λ) ≤ p^T c = cᵀp

So: dual objective ≤ primal objective at every feasible point.

**Strong duality** (true for this LP, since it's always feasible and bounded):
at the optimum, primal = dual.

    min_p cᵀp  =  max_λ b_eq^T λ

This is the central fact. Now let's unpack what it means for OT.

---

## 2. Deriving the OT Dual

Recall the primal variables and constraints from Phase 1:

    p ∈ ℝⁿᵐ          (flattened transport plan, p[i*m+j] = Pᵢⱼ)
    c ∈ ℝⁿᵐ          (flattened cost matrix)
    A_eq ∈ ℝ^{(n+m)×nm}
    b_eq = [a; b]

We split the dual variable λ ∈ ℝⁿ⁺ᵐ into two parts:

    u ∈ ℝⁿ    (one variable per source — the "source prices")
    v ∈ ℝᵐ    (one variable per target — the "target prices")

The dual constraint A_eq^T λ ≤ c says, for each variable p[i*m+j] = Pᵢⱼ:

    [row i of A_eq^T corresponding to Pᵢⱼ] · [u; v]  ≤  Cᵢⱼ

What is row (i*m+j) of A_eq^T? It's column (i*m+j) of A_eq.
From Phase 1: column (i*m+j) of A_eq has a 1 in row i (source constraint)
and a 1 in row n+j (target constraint), zeros elsewhere.

So the dual constraint for the (i,j) pair is:

    uᵢ + vⱼ ≤ Cᵢⱼ       for all i ∈ {1,…,n}, j ∈ {1,…,m}

The dual objective b_eq^T λ = aᵀu + bᵀv.

**The dual problem:**

    max_{u ∈ ℝⁿ, v ∈ ℝᵐ}   aᵀu + bᵀv
    s.t.                     uᵢ + vⱼ ≤ Cᵢⱼ   for all i, j

That's it. The whole dual. n+m variables instead of nm.
For a 100×100 problem: primal has 10,000 variables, dual has only 200.

---

## 3. Economic Interpretation

Imagine you are a shipping company. You set:
- uᵢ = price to pick up one unit at source i
- vⱼ = price to deliver one unit to target j

The customer pays uᵢ + vⱼ to ship from i to j.

**The dual constraint** uᵢ + vⱼ ≤ Cᵢⱼ says: you cannot charge more than the actual
transport cost. If you did, the customer would just drive it themselves.

**The dual objective** aᵀu + bᵀv = total revenue.

The dual asks: **what's the maximum revenue a shipping company can earn, subject to
not being undercut by self-transport?**

Strong duality says: at the optimum, this maximum revenue equals the minimum
transport cost. The two problems have the same value.

---

## 4. Complementary Slackness

This is the most powerful consequence of duality. At optimality:

    Pᵢⱼ · (Cᵢⱼ − uᵢ − vⱼ) = 0     for all i, j

Two cases:
- If Pᵢⱼ > 0 (mass is transported along route (i,j)):  Cᵢⱼ = uᵢ + vⱼ
  The shipping price exactly equals the transport cost. This route "breaks even."
- If Cᵢⱼ > uᵢ + vⱼ (the route is "overpriced" relative to actual cost):  Pᵢⱼ = 0
  No mass flows along this route.

In words: **mass only flows along routes where the price exactly covers the cost.**
Expensive routes (where Cᵢⱼ > uᵢ + vⱼ) are never used.

This will become crucial in Phase 3: the Sinkhorn algorithm enforces a "softened"
version of this condition.

---

## 5. The c-Transform

Given any u ∈ ℝⁿ, the **c-transform** of u is the vector v = u^c ∈ ℝᵐ defined by:

    (u^c)ⱼ = min_i (Cᵢⱼ − uᵢ)

This is the tightest v that satisfies the dual constraint uᵢ + vⱼ ≤ Cᵢⱼ.
It says: "given source prices u, what is the maximum price you can charge at target j?"

Starting from any u, we can always find the best v by applying the c-transform.
Then apply it again to get the best u given v. This alternating update is essentially
the continuous version of Sinkhorn (more on that in Phase 3).

---

## 6. Worked Example — 2×2

Using the same example from Phase 1:

```
a = [0.6, 0.4],  b = [0.5, 0.5]

C = [[1, 3],
     [2, 1]]

Primal optimal: P* = [[0.5, 0.1], [0.0, 0.4]],  cost = 1.2
```

**What are the dual variables (u, v) at optimum?**

Complementary slackness: active routes (Pᵢⱼ > 0) must satisfy uᵢ + vⱼ = Cᵢⱼ.

Active routes from P*: (1,1), (1,2), (2,2). So three equations:

    u₁ + v₁ = C₁₁ = 1    ...(A)
    u₁ + v₂ = C₁₂ = 3    ...(B)
    u₂ + v₂ = C₂₂ = 1    ...(C)

Three equations, four unknowns (u₁, u₂, v₁, v₂). The system is underdetermined
by one degree of freedom — dual variables are only determined up to a constant shift.
(Adding δ to all uᵢ and subtracting δ from all vⱼ leaves the constraints and objective
unchanged. This is the LP redundancy from Phase 1.)

Fix u₁ = 0 (our free choice):
    (A): v₁ = 1
    (B): v₂ = 3
    (C): u₂ = 1 - v₂ = 1 - 3 = -2

**Dual variables: u = [0, -2],  v = [1, 3]**

**Verify the dual constraint** uᵢ + vⱼ ≤ Cᵢⱼ for all (i,j):

    (1,1): 0 + 1 = 1 ≤ 1 = C₁₁  ✓  (tight — route is used)
    (1,2): 0 + 3 = 3 ≤ 3 = C₁₂  ✓  (tight — route is used)
    (2,1): -2 + 1 = -1 ≤ 2 = C₂₁  ✓  (slack — route not used, P₂₁=0)
    (2,2): -2 + 3 = 1 ≤ 1 = C₂₂  ✓  (tight — route is used)

**Verify strong duality:**

    Dual objective = aᵀu + bᵀv
                   = 0.6·0 + 0.4·(-2)  +  0.5·1 + 0.5·3
                   = 0     − 0.8        +  0.5   + 1.5
                   = 1.2

Primal cost = 1.2. ✓ Strong duality holds.

---

## 7. The 3×3 Example

```
a = [0.5, 0.3, 0.2],  b = [0.4, 0.4, 0.2]

C = [[1, 2, 4],
     [3, 1, 2],
     [4, 3, 1]]

Primal optimal: P* = [[0.4, 0.1, 0], [0, 0.3, 0], [0, 0, 0.2]],  cost = 1.1
```

Active routes: (1,1), (1,2), (2,2), (3,3).

Equations from complementary slackness:
    u₁ + v₁ = 1    ...(A)
    u₁ + v₂ = 2    ...(B)
    u₂ + v₂ = 1    ...(C)
    u₃ + v₃ = 1    ...(D)

Four equations, six unknowns. Two free parameters.
Fix u₁ = 0, v₃ = 0:
    (A): v₁ = 1
    (B): v₂ = 2
    (C): u₂ = 1 - 2 = -1
    (D): u₃ = 1

**u = [0, -1, 1],  v = [1, 2, 0]**

Verify dual objective = aᵀu + bᵀv:
    = 0.5·0 + 0.3·(-1) + 0.2·1  +  0.4·1 + 0.4·2 + 0.2·0
    = 0     − 0.3       + 0.2    +  0.4   + 0.8   + 0
    = 1.1  ✓

---

## 8. Key Takeaways

| Concept | What it means |
|---|---|
| Dual variables (u, v) | Prices at each source/target |
| Dual constraint uᵢ+vⱼ ≤ Cᵢⱼ | Prices can't exceed actual cost |
| Dual objective aᵀu + bᵀv | Total revenue at these prices |
| Strong duality | Primal min = Dual max at optimum |
| Complementary slackness | Mass flows only on tight routes (uᵢ+vⱼ = Cᵢⱼ) |
| c-transform | Given u, the tightest feasible v: vⱼ = minᵢ(Cᵢⱼ−uᵢ) |
| Degree of freedom | Dual solution unique only up to a constant shift |

**What's next:** Duality unlocks the Sinkhorn algorithm in Phase 3. Sinkhorn solves
a regularized version of the dual — it enforces the constraints "softly" via
exponentials instead of hard inequalities. The c-transform becomes a log-sum-exp.
