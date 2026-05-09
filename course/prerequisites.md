# Prerequisites — Math Primer

*Everything you need to read this course. Takes about 30 minutes.*

---

## 1. Probability Distributions

A **probability distribution** describes where random things tend to land.

**Intuition:** Roll a die 1000 times and plot how often each number came up. That histogram is an empirical distribution.

**The key objects:**
- **Discrete distribution:** a list of values and their probabilities. E.g. `{1: 0.2, 2: 0.5, 3: 0.3}` means value 1 has probability 0.2, etc. Probabilities must sum to 1.
- **Continuous distribution:** a probability density function (PDF) p(x), where the area under the curve equals 1. Think of it as the limit of a histogram with infinitely thin bars.
- **Samples:** random points drawn from a distribution. If you sample 1000 times from p(x), the histogram of samples approximates p(x).

**In this course:** we mostly work with samples — point clouds of 2D coordinates. A "distribution" is often just an array of points.

```
Discrete:                          Continuous:
values:     1   2   3              PDF:   ▁▄█▄▁
prob:      0.2 0.5 0.3                     x
```

---

## 2. Marginals

Given a table of joint probabilities P[i,j] (probability that X=i AND Y=j), the **marginal** of X is obtained by summing over all values of Y:

```
P_X(i) = Σ_j P[i,j]    (sum each row)
P_Y(j) = Σ_i P[i,j]    (sum each column)
```

**Why it matters for OT:** A transport plan P[i,j] says "how much mass flows from source i to target j." The marginals of P must match the source distribution (sum of each row = how much source i has) and the target distribution (sum of each column = how much target j needs). This is the key constraint in the OT problem.

---

## 3. Linear Programs

A **linear program (LP)** is an optimisation problem where:
- The objective (what you minimise or maximise) is linear in the variables
- The constraints are linear equalities or inequalities

**Example:**
```
Minimise:    3x + 2y
Subject to:  x + y  ≥ 4
             x      ≥ 1
             y      ≥ 0
```

Linear programs have a beautiful property: if a solution exists, the optimal solution is always at a *corner* (vertex) of the feasible region. This means they can be solved exactly and efficiently.

**The OT problem as an LP:**
- Variables: P[i,j] for all source-target pairs — the entries of the transport plan
- Objective: minimise `Σ_{i,j} C[i,j] * P[i,j]` (total cost)
- Constraints: P[i,:].sum() = a[i] for all i (row marginals = source), P[:,j].sum() = b[j] for all j (column marginals = target), P[i,j] ≥ 0 (non-negative)

This is exactly what `scipy.optimize.linprog` solves.

---

## 4. LP Duality

Every LP has a "mirror image" called its **dual**. If the original (primal) problem is:

```
Minimise:   cᵀx
Subject to: Ax = b, x ≥ 0
```

The dual is:
```
Maximise:   bᵀy
Subject to: Aᵀy ≤ c
```

**Key theorem — Strong Duality:** The optimal values of the primal and dual are equal. `min cᵀx = max bᵀy`.

**What this means for OT:** The dual of the transport LP has a beautiful interpretation. The dual variables (u[i], v[j]) are "prices" — u[i] is the value of having one unit of mass at source i, v[j] is the value of that mass arriving at target j. They satisfy `u[i] + v[j] ≤ C[i,j]` for all pairs, and the optimal prices achieve equality wherever mass actually flows.

---

## 5. KL Divergence

**Kullback-Leibler divergence** measures how much one distribution differs from another.

For discrete distributions p and q:
```
KL(p ‖ q) = Σ_i p_i · log(p_i / q_i)
```

**Properties:**
- KL(p ‖ q) ≥ 0 always
- KL(p ‖ q) = 0 if and only if p = q
- **Not symmetric:** KL(p ‖ q) ≠ KL(q ‖ p) in general
- KL(p ‖ q) = ∞ if q_i = 0 anywhere that p_i > 0

**Why KL fails for comparing distributions with non-overlapping support:**
If p is concentrated at position 0 and q is concentrated at position 1, KL(p ‖ q) = ∞. Two distributions that are "almost the same but shifted by a tiny amount" can have infinite KL divergence. This is why OT (which uses distance, not log-ratio) is more useful for many applications.

**Why KL appears in Sinkhorn (Phase 3):** The Sinkhorn algorithm solves a *regularised* OT problem where we add an entropy penalty: `min ⟨C, P⟩ + ε · KL(P ‖ a⊗b)`. The KL penalty encourages the transport plan to be "spread out" rather than concentrated, which makes the problem strictly convex and easier to solve iteratively.

---

## 6. Vectors and Matrices

You need to be comfortable with:

**Dot product:** `⟨a, b⟩ = Σ_i a_i * b_i` — measures alignment between two vectors.

**Matrix-vector product:** `(Av)_i = Σ_j A_{ij} v_j` — each output is a weighted sum of inputs.

**Element-wise operations:** In NumPy, `A * B` multiplies element-by-element. `A @ B` is matrix multiplication.

**Broadcasting:** `A / a[:, None]` divides each row of matrix A by the corresponding entry of vector a. The `[:,None]` reshapes a 1D vector into a column so NumPy can broadcast.

**Tensor notation:** In this course we often write shapes explicitly. `(n, m)` means a 2D array with n rows and m columns. `(n,)` means a 1D vector of length n.

---

## 7. Gradients and Optimisation

A **gradient** is a vector of partial derivatives. For a function f(x₁, x₂, ..., xₙ), the gradient ∇f is the vector `[∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`.

The gradient points in the direction of steepest increase. To minimise f, move in the **negative gradient direction** — this is gradient descent:

```
x ← x - α · ∇f(x)    where α is the step size (learning rate)
```

**For neural networks (Phase 6):** PyTorch computes gradients automatically via backpropagation. You define the loss, call `loss.backward()`, and gradients accumulate in `param.grad` for each parameter. Then `optimizer.step()` updates parameters.

---

## 8. What Is a Neural Network?

A **multilayer perceptron (MLP)** applies alternating linear transformations and nonlinear activations:

```
h₁ = activation(W₁ · x + b₁)
h₂ = activation(W₂ · h₁ + b₂)
y  = W₃ · h₂ + b₃
```

The weights W and biases b are the learnable parameters. Training finds values that minimise a loss function.

**SiLU activation:** `SiLU(x) = x · σ(x)` where σ(x) = 1/(1 + e^{-x}). Smooth and non-monotone — better than ReLU for fitting smooth functions like the velocity fields in Phase 6.

**In Phase 6:** the network takes a 2D position and a time t as input, and predicts a 2D velocity. Its job: for any point in space and any moment in time, tell you which direction to move to arrive at a data sample.

---

## 9. Exponential and Log

The exponential `exp(x) = eˣ` and natural logarithm `log(x) = ln(x)` appear constantly. Key identities:

```
log(a · b)   = log(a) + log(b)
log(a / b)   = log(a) - log(b)
exp(log(x))  = x
log(exp(x))  = x
log(Σᵢ exp(aᵢ))  ← this is log-sum-exp, computed stably as: max(a) + log(Σᵢ exp(aᵢ - max(a)))
```

**Why log-sum-exp stability matters (Phase 3):** In the Sinkhorn algorithm, we compute `exp(-C/ε)` where C can be large and ε is small. This can overflow to infinity or underflow to zero. The log-domain trick keeps all computations in log space to avoid this.

---

## 10. What Is a Cumulative Distribution Function (CDF)?

For a distribution with values x₁ < x₂ < ... < xₙ and probabilities p₁, p₂, ..., pₙ, the **CDF** at value x is:

```
F(x) = Σ_{xᵢ ≤ x} pᵢ    (total probability to the left of x)
```

F is a non-decreasing step function that starts at 0 and ends at 1.

The **quantile function** (inverse CDF) Q(u) answers: what value x has CDF equal to u?

```
Q(u) = min { x : F(x) ≥ u }
```

**Why this matters for W₁ and W₂ in 1D (Phase 4):** In one dimension, the optimal transport between two distributions p and q is just the integral of |Q_p(u) - Q_q(u)|^p du — comparing quantile functions pointwise. Sorting both arrays and computing differences is equivalent.

---

## Quick Reference Table

| Symbol | Meaning |
|--------|---------|
| μ, ν   | probability distributions (mu, nu) |
| p, q   | discrete probability vectors (sum to 1) |
| P      | transport plan (matrix, rows = source, cols = target) |
| C      | cost matrix, C[i,j] = cost to move mass from i to j |
| ε      | regularisation strength (epsilon, Greek letter) |
| ⟨A,B⟩ | sum of element-wise products: Σ_{i,j} A[i,j]·B[i,j] |
| _# μ  | pushforward: apply a map to a distribution (see glossary) |
| KL(p‖q) | KL divergence from p to q |
| W₂    | Wasserstein-2 distance |
| ∇     | gradient (del) |
| argmin | the value of x that minimises f(x) |

---

## You're Ready

If you understood most of the above — even if some parts felt vague — you have enough to follow the course. Concepts will be re-explained in context as they come up.

Start with `course/01_discrete_ot.md`.
