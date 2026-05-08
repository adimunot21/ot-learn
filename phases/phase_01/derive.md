# Phase 1 Derivation — Discrete Optimal Transport as a Linear Program

## The Problem in One Sentence

Given two piles of mass and a cost to move mass between locations, find the cheapest
way to rearrange the first pile into the second.

---

## 1. Setting Up Notation

We have:
- **n source locations** x₁, …, xₙ with weights **a ∈ ℝⁿ**, aᵢ ≥ 0, Σᵢ aᵢ = 1
- **m target locations** y₁, …, yₘ with weights **b ∈ ℝᵐ**, bⱼ ≥ 0, Σⱼ bⱼ = 1
- **Cost matrix C ∈ ℝⁿˣᵐ**, where Cᵢⱼ = cost of moving one unit of mass from xᵢ to yⱼ

Think of it concretely: 3 bakeries (sources) supply bread to 3 cafes (targets).
aᵢ = fraction of total bread supply at bakery i.
bⱼ = fraction of total demand at cafe j.
Cᵢⱼ = distance (km) between bakery i and cafe j.

---

## 2. Monge's Formulation (1781)

Gaspard Monge framed this as: find a **map** T: {x₁,…,xₙ} → {y₁,…,yₘ} such that:

1. **Mass conservation**: mass assigned to each target is correct.
   Formally: for each j, Σᵢ aᵢ · 𝟙[T(xᵢ) = yⱼ] = bⱼ

2. **Minimum total cost**: min_T Σᵢ aᵢ · C(xᵢ, T(xᵢ))

**The fatal flaw:** a map can only send each source to *one* target.

Example: a = [1.0], b = [0.5, 0.5], C = [[1, 2]]
Source x₁ holds all the mass. We need half to go to y₁, half to y₂.
No map T can do this — T(x₁) can only equal y₁ or y₂, not both.

Monge's problem is **infeasible whenever it's impossible to split mass**.
This is a fundamental limitation: maps are rigid, real transport is not.

---

## 3. Kantorovich's Relaxation (1942)

Leonid Kantorovich's key insight: drop the constraint that mass moves as a rigid map.
Instead, describe transport as a **joint distribution** over (source, target) pairs.

Define the **transport plan** P ∈ ℝⁿˣᵐ where:

    Pᵢⱼ = amount of mass moved from source i to target j

Three constraints:

```
(i)  Pᵢⱼ ≥ 0                    for all i, j    (can't move negative mass)
(ii) Σⱼ Pᵢⱼ = aᵢ               for all i        (must ship all of source i)
(iii) Σᵢ Pᵢⱼ = bⱼ              for all j        (must fill all of target j)
```

Constraints (ii) and (iii) are called the **marginal constraints**.
The set of all feasible P is called the **transport polytope** U(a, b).

Objective: minimize total cost

    min_{P ∈ U(a,b)}  Σᵢⱼ Cᵢⱼ Pᵢⱼ  =  min_{P ∈ U(a,b)}  ⟨C, P⟩_F

where ⟨·,·⟩_F is the Frobenius (elementwise) inner product.

**Why this always has a solution:** U(a,b) is always non-empty.
The product measure P = a bᵀ (outer product) always satisfies the marginal constraints:
  row i sum = aᵢ · Σⱼ bⱼ = aᵢ · 1 = aᵢ  ✓
  col j sum = (Σᵢ aᵢ) · bⱼ = 1 · bⱼ = bⱼ  ✓
So U(a,b) is non-empty, and since it's a bounded polyhedron, the LP minimum exists.

**Monge as a special case:** a Monge map T corresponds to a P where each row has at
most one non-zero entry (all mass from source i goes to exactly one target T(xᵢ)).
These are extreme points of U(a,b) — valid, but not all extreme points are Monge maps.

---

## 4. This Is a Linear Program

Let's write it in standard LP form.

**Step 1 — Vectorize P.**
Flatten P row-by-row: p = P.flatten() ∈ ℝⁿᵐ, so p[i·m + j] = Pᵢⱼ.
Similarly flatten C: c = C.flatten() ∈ ℝⁿᵐ.

The objective ⟨C, P⟩_F = cᵀp (just a dot product after flattening).

**Step 2 — Write the marginal constraints as A_eq p = b_eq.**

Source marginal i (row sum of P equals aᵢ):
    Σⱼ p[i·m + j] = aᵢ
→ Row i of A_src has 1s at positions {i·m, i·m+1, …, i·m+(m-1)} and 0s elsewhere.
    A_src ∈ ℝⁿˣⁿᵐ

Target marginal j (col sum of P equals bⱼ):
    Σᵢ p[i·m + j] = bⱼ
→ Row j of A_tgt has 1s at positions {j, m+j, 2m+j, …, (n-1)·m+j} and 0s elsewhere.
    A_tgt ∈ ℝᵐˣⁿᵐ

Stack them:

    A_eq = [ A_src ]   ∈ ℝ^{(n+m) × nm}
           [ A_tgt ]

    b_eq = [ a ]       ∈ ℝⁿ⁺ᵐ
           [ b ]

**Step 3 — Bounds.** p[k] ∈ [0, ∞) for all k.

**Final LP:**

    min   cᵀ p
    s.t.  A_eq p = b_eq
          p ≥ 0

This has nm variables, n+m equality constraints (but only n+m−1 are independent
since Σᵢ aᵢ = Σⱼ bⱼ = 1 makes one constraint redundant), and is guaranteed feasible.

---

## 5. Worked Example — 2×2 by Hand

**Setup:**
```
a = [0.6, 0.4]      (source weights)
b = [0.5, 0.5]      (target weights)

C = [[1, 3],        C₁₁=1  C₁₂=3
     [2, 1]]        C₂₁=2  C₂₂=1
```

**Variables:** P₁₁, P₁₂, P₂₁, P₂₂ (4 variables, 3 independent constraints + bound)

**Constraints:**
```
P₁₁ + P₁₂        = 0.6    (source 1 ships out all its mass)
       P₂₁ + P₂₂ = 0.4    (source 2 ships out all its mass)
P₁₁       + P₂₁  = 0.5    (target 1 receives exactly its demand)
       P₁₂ + P₂₂ = 0.5    (target 2 receives exactly its demand)
Pᵢⱼ ≥ 0
```

**Solving by inspection:**
Cost 1 paths are cheap: C₁₁=1, C₂₂=1. Cost 3 path C₁₂=3 is expensive.
Strategy: maximize P₁₁ and P₂₂ (the cheap ones).

- P₁₁ ≤ min(a₁, b₁) = min(0.6, 0.5) = 0.5 → set P₁₁ = 0.5
- Remaining from source 1: a₁ - P₁₁ = 0.6 - 0.5 = 0.1 → P₁₂ = 0.1
- Remaining for target 1: b₁ - P₁₁ = 0.5 - 0.5 = 0.0 → P₂₁ = 0.0
- Then P₂₂ = a₂ - P₂₁ = 0.4 - 0.0 = 0.4

**Solution:**
```
P* = [[0.5, 0.1],
      [0.0, 0.4]]
```

**Verify marginals:**
```
Row sums: [0.5+0.1, 0.0+0.4] = [0.6, 0.4] = a  ✓
Col sums: [0.5+0.0, 0.1+0.4] = [0.5, 0.5] = b  ✓
```

**Optimal cost:**
```
⟨C, P*⟩ = 0.5·1 + 0.1·3 + 0.0·2 + 0.4·1
         = 0.5  + 0.3  + 0.0  + 0.4
         = 1.2
```

Note: the "trivial" plan P = a bᵀ = [[0.3, 0.3], [0.2, 0.2]] costs:
0.3·1 + 0.3·3 + 0.2·2 + 0.2·1 = 0.3 + 0.9 + 0.4 + 0.2 = 1.8
Our plan saves 0.6 in transport cost — it routes mass intelligently.

---

## 6. The 3×3 Example for Implementation

**Setup (bakeries → cafes, distances in km):**
```
a = [0.5, 0.3, 0.2]     (supply fractions)
b = [0.4, 0.4, 0.2]     (demand fractions)

C = [[1, 2, 4],          close   medium  far
     [3, 1, 2],          far     close   medium
     [4, 3, 1]]          far     medium  close
```

The cost matrix is designed so diagonal entries are cheap (same "neighborhood"),
off-diagonal entries increase with distance from the diagonal.

**Greedy trace (not necessarily optimal, but gives intuition):**
```
Greedy: always ship on the cheapest available path.

Step 1: cheapest cells are C₁₁=1, C₂₂=1, C₃₃=1.
  P₁₁ = min(a₁, b₁) = min(0.5, 0.4) = 0.4   → remaining: a₁=0.1, b₁=0.0
  P₂₂ = min(a₂, b₂) = min(0.3, 0.4) = 0.3   → remaining: a₂=0.0, b₂=0.1
  P₃₃ = min(a₃, b₃) = min(0.2, 0.2) = 0.2   → remaining: a₃=0.0, b₃=0.0

Step 2: unmet: source 1 still has 0.1 to ship. Target 2 still needs 0.1.
  P₁₂ = 0.1   (cost C₁₂=2, the next cheapest option for source 1)
```

**Greedy solution:**
```
P_greedy = [[0.4, 0.1, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.2]]

Cost = 0.4·1 + 0.1·2 + 0.3·1 + 0.2·1 = 0.4 + 0.2 + 0.3 + 0.2 = 1.1
```

We expect the LP to find cost ≤ 1.1. (It may match or beat it.)

---

## 7. Key Takeaways

| Concept | What it means |
|---|---|
| Monge map T | Rigid: each source maps to exactly one target |
| Kantorovich plan P | Flexible: mass can be split across targets |
| U(a, b) | The transport polytope — convex set of all feasible plans |
| ⟨C, P⟩ | The total transport cost (what we minimize) |
| LP formulation | nm variables, n+m constraints, always feasible |
| Product measure a⊗b | Baseline: independent coupling, usually not optimal |

**What's next:** The LP has a dual problem. The dual variables have a beautiful
economic interpretation (shipping prices), and strong duality tells us something
deep about the structure of optimal transport plans. That's Phase 2.
