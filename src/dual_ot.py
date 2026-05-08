"""
Kantorovich dual of discrete OT.

The dual problem:
    max_{u, v}  a^T u + b^T v
    s.t.        u_i + v_j <= C_ij   for all i, j

Solved by feeding the dual LP into scipy.optimize.linprog.
We verify strong duality (primal == dual) and complementary slackness.

Math: phases/phase_02/derive.md
"""

import time
import numpy as np
from scipy.optimize import linprog

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lp_ot import solve_ot, transport_cost

CONFIG = {
    "lp_method": "highs",
    "lp_options": {"disp": False},
    "complementary_slackness_tol": 1e-6,
    "strong_duality_tol": 1e-5,
}


# ---------------------------------------------------------------------------
# Dual solver
# ---------------------------------------------------------------------------

def solve_dual_ot(
    cost_matrix: np.ndarray,      # (n, m)
    source_weights: np.ndarray,   # (n,)
    target_weights: np.ndarray,   # (m,)
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the Kantorovich dual:
        max  a^T u + b^T v
        s.t. u_i + v_j <= C_ij   for all i, j

    Converts to standard LP min form:
        min  -a^T u - b^T v
        s.t. u_i + v_j <= C_ij   (n*m inequality constraints)

    Variables: w = [u; v] ∈ ℝ^{n+m}, unbounded.

    Returns
    -------
    u            : ndarray (n,)   source dual potentials
    v            : ndarray (m,)   target dual potentials
    dual_value   : float          a^T u + b^T v at optimum
    """
    n, m = cost_matrix.shape

    # Objective: minimise -a^T u - b^T v  →  coefficients [-a; -b]
    c_dual = np.concatenate([-source_weights, -target_weights])  # (n+m,)

    # Inequality constraints: u_i + v_j <= C_ij for all (i,j)
    # Variable layout: w[0..n-1] = u,  w[n..n+m-1] = v
    # One row per (i,j) pair:  e_i^T [I_n | 0] + e_j^T [0 | I_m]
    n_constraints = n * m
    A_ub = np.zeros((n_constraints, n + m))  # (n*m, n+m)
    b_ub = cost_matrix.flatten()             # (n*m,)

    for i in range(n):
        for j in range(m):
            row = i * m + j
            A_ub[row, i]     = 1.0   # coefficient of u_i
            A_ub[row, n + j] = 1.0   # coefficient of v_j

    # Dual variables are unbounded (no box constraints)
    bounds = [(None, None)] * (n + m)

    result = linprog(
        c_dual,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method=CONFIG["lp_method"],
        options=CONFIG["lp_options"],
    )

    if result.status != 0:
        raise RuntimeError(f"Dual LP solver failed: {result.message}")

    u = result.x[:n]       # (n,)
    v = result.x[n:]       # (m,)
    dual_value = float(-result.fun)   # negate because we minimised the negative
    return u, v, dual_value


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def c_transform(u: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the c-transform of u: v_j = min_i (C_ij - u_i)

    Given source potentials u, returns the tightest feasible target potentials v.

    u            : (n,)
    cost_matrix  : (n, m)
    returns v    : (m,)
    """
    # cost_matrix - u[:, None] broadcasts to (n, m), then min over axis 0
    return (cost_matrix - u[:, None]).min(axis=0)   # (n,m) → (m,)


def check_complementary_slackness(
    transport_plan: np.ndarray,   # (n, m)
    u: np.ndarray,                # (n,)
    v: np.ndarray,                # (m,)
    cost_matrix: np.ndarray,      # (n, m)
    tol: float = CONFIG["complementary_slackness_tol"],
) -> bool:
    """
    Verify: P_ij > 0  =>  u_i + v_j == C_ij  (up to tolerance).

    Returns True if all active routes satisfy the condition.
    """
    slack = cost_matrix - u[:, None] - v[None, :]    # (n,m)  C_ij - u_i - v_j
    active = transport_plan > tol
    violations = active & (slack > tol)
    return not violations.any()


# ---------------------------------------------------------------------------
# Sanity checks / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("KANTOROVICH DUALITY — SANITY CHECKS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: 2×2 hand-traced example
    # ------------------------------------------------------------------
    print("\n--- Test 1: 2×2 hand-traced example ---")
    a = np.array([0.6, 0.4])
    b = np.array([0.5, 0.5])
    C = np.array([[1.0, 3.0],
                  [2.0, 1.0]])

    P, primal_cost = solve_ot(C, a, b)
    u, v, dual_value = solve_dual_ot(C, a, b)

    print(f"Primal cost:   {primal_cost:.6f}")
    print(f"Dual value:    {dual_value:.6f}  (must equal primal)")
    print(f"u (source potentials): {np.round(u, 4)}")
    print(f"v (target potentials): {np.round(v, 4)}")

    # Strong duality
    gap = abs(primal_cost - dual_value)
    print(f"Duality gap:   {gap:.2e}  (expected ~0)")
    assert gap < CONFIG["strong_duality_tol"], "Strong duality violated"

    # Dual constraint: u_i + v_j <= C_ij everywhere
    slack = C - u[:, None] - v[None, :]         # (2,2)
    print(f"Dual slack (C - u - v):\n{np.round(slack, 4)}  (all must be >= 0)")
    assert (slack >= -1e-8).all(), "Dual constraint violated"

    # Complementary slackness
    cs_ok = check_complementary_slackness(P, u, v, C)
    print(f"Complementary slackness: {'PASS' if cs_ok else 'FAIL'}")
    assert cs_ok

    # c-transform check: applying c-transform to u should recover v (up to shift)
    v_ctransform = c_transform(u, C)            # (2,)
    shift = (v - v_ctransform).mean()
    print(f"c-transform recovers v (up to shift {shift:.4f}): "
          f"{'PASS' if np.allclose(v - shift, v_ctransform, atol=1e-5) else 'FAIL'}")

    print("PASS")

    # ------------------------------------------------------------------
    # Test 2: 3×3 example
    # ------------------------------------------------------------------
    print("\n--- Test 2: 3×3 example ---")
    a3 = np.array([0.5, 0.3, 0.2])
    b3 = np.array([0.4, 0.4, 0.2])
    C3 = np.array([[1.0, 2.0, 4.0],
                   [3.0, 1.0, 2.0],
                   [4.0, 3.0, 1.0]])

    t0 = time.perf_counter()
    P3, primal_cost3 = solve_ot(C3, a3, b3)
    u3, v3, dual_value3 = solve_dual_ot(C3, a3, b3)
    elapsed = time.perf_counter() - t0

    gap3 = abs(primal_cost3 - dual_value3)
    print(f"Primal cost:   {primal_cost3:.6f}")
    print(f"Dual value:    {dual_value3:.6f}")
    print(f"Duality gap:   {gap3:.2e}")
    print(f"u = {np.round(u3, 4)}")
    print(f"v = {np.round(v3, 4)}")
    print(f"Solve time:    {elapsed*1000:.2f} ms")

    assert gap3 < CONFIG["strong_duality_tol"]
    assert check_complementary_slackness(P3, u3, v3, C3)
    assert (C3 - u3[:, None] - v3[None, :] >= -1e-8).all()
    print("PASS")

    # ------------------------------------------------------------------
    # Test 3: random instances — strong duality at scale
    # ------------------------------------------------------------------
    print("\n--- Test 3: 50 random instances, strong duality check ---")
    rng = np.random.default_rng(0)
    max_gap = 0.0
    for _ in range(50):
        n_r = rng.integers(2, 10)
        m_r = rng.integers(2, 10)
        a_r = rng.dirichlet(np.ones(n_r))
        b_r = rng.dirichlet(np.ones(m_r))
        C_r = rng.uniform(0, 10, size=(n_r, m_r))
        _, pc = solve_ot(C_r, a_r, b_r)
        _, _, dv = solve_dual_ot(C_r, a_r, b_r)
        max_gap = max(max_gap, abs(pc - dv))

    print(f"Max duality gap across 50 trials: {max_gap:.2e}")
    assert max_gap < CONFIG["strong_duality_tol"]
    print("PASS")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
