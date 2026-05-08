"""
Discrete Optimal Transport solved as a Linear Program.

Math: phases/phase_01/derive.md
"""

import time
import numpy as np
from scipy.optimize import linprog

CONFIG = {
    "lp_method": "highs",   # HiGHS is the default LP solver in scipy >= 1.9; fastest available
    "lp_options": {"disp": False},
}


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def build_constraint_matrix(n: int, m: int) -> np.ndarray:
    """
    Build A_eq for the OT linear program.

    Variables are p = P.flatten() (row-major), shape (n*m,).
    Returns A_eq of shape (n+m, n*m).

    Rows 0..n-1   enforce source marginals: sum_j P[i,j] = a[i]
    Rows n..n+m-1 enforce target marginals: sum_i P[i,j] = b[j]
    """
    nm = n * m
    A_eq = np.zeros((n + m, nm))

    # Source marginals: row i of A_eq has 1s at columns {i*m, i*m+1, ..., i*m+(m-1)}
    for i in range(n):
        A_eq[i, i * m : i * m + m] = 1.0  # shape trace: A_eq[i] ∈ ℝ^{nm}

    # Target marginals: row n+j of A_eq has 1s at columns {j, m+j, 2m+j, ..., (n-1)*m+j}
    for j in range(m):
        A_eq[n + j, j::m] = 1.0  # shape trace: stride-m slice of ℝ^{nm}

    return A_eq  # (n+m, n*m)


def solve_ot(
    cost_matrix: np.ndarray,      # (n, m)
    source_weights: np.ndarray,   # (n,)  must sum to 1
    target_weights: np.ndarray,   # (m,)  must sum to 1
) -> tuple[np.ndarray, float]:
    """
    Solve discrete OT:  min  <C, P>  s.t.  P @ 1 = a,  P.T @ 1 = b,  P >= 0

    Returns
    -------
    transport_plan : ndarray of shape (n, m)
    optimal_cost   : float
    """
    n, m = cost_matrix.shape
    assert source_weights.shape == (n,), f"expected source_weights shape ({n},)"
    assert target_weights.shape == (m,), f"expected target_weights shape ({m},)"
    np.testing.assert_allclose(
        source_weights.sum(), 1.0, atol=1e-6, err_msg="source_weights must sum to 1"
    )
    np.testing.assert_allclose(
        target_weights.sum(), 1.0, atol=1e-6, err_msg="target_weights must sum to 1"
    )

    c = cost_matrix.flatten()                          # (n*m,) objective coefficients
    A_eq = build_constraint_matrix(n, m)               # (n+m, n*m)
    b_eq = np.concatenate([source_weights,             # (n+m,)
                           target_weights])
    bounds = [(0.0, None)] * (n * m)                   # Pᵢⱼ ≥ 0

    result = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=CONFIG["lp_method"],
        options=CONFIG["lp_options"],
    )

    if result.status != 0:
        raise RuntimeError(f"LP solver failed: {result.message}")

    transport_plan = result.x.reshape(n, m)   # (n, m)
    optimal_cost = float(result.fun)
    return transport_plan, optimal_cost


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def marginal_error(
    transport_plan: np.ndarray,    # (n, m)
    source_weights: np.ndarray,    # (n,)
    target_weights: np.ndarray,    # (m,)
) -> tuple[float, float]:
    """Max absolute deviation from the marginal constraints."""
    source_err = np.abs(transport_plan.sum(axis=1) - source_weights).max()
    target_err = np.abs(transport_plan.sum(axis=0) - target_weights).max()
    return float(source_err), float(target_err)


def transport_cost(cost_matrix: np.ndarray, transport_plan: np.ndarray) -> float:
    """Compute <C, P> (Frobenius inner product)."""
    return float(np.sum(cost_matrix * transport_plan))  # (n,m) ⊙ (n,m) → scalar


# ---------------------------------------------------------------------------
# Sanity checks / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DISCRETE OT — LP SOLVER SANITY CHECKS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: 2×2 hand-traced example from derive.md
    # ------------------------------------------------------------------
    print("\n--- Test 1: 2×2 hand-traced example ---")
    a_2 = np.array([0.6, 0.4])
    b_2 = np.array([0.5, 0.5])
    C_2 = np.array([[1.0, 3.0],
                    [2.0, 1.0]])

    P_2, cost_2 = solve_ot(C_2, a_2, b_2)
    src_err, tgt_err = marginal_error(P_2, a_2, b_2)

    print(f"Transport plan P:\n{P_2}")
    print(f"Optimal cost:  {cost_2:.4f}  (expected 1.2000)")
    print(f"Marginal error — source: {src_err:.2e}  target: {tgt_err:.2e}")
    assert abs(cost_2 - 1.2) < 1e-4, "2×2 cost mismatch"
    assert src_err < 1e-6 and tgt_err < 1e-6, "marginal constraints violated"
    print("PASS")

    # ------------------------------------------------------------------
    # Test 2: 3×3 example from derive.md (greedy cost was 1.1)
    # ------------------------------------------------------------------
    print("\n--- Test 2: 3×3 bakery-cafe example ---")
    a_3 = np.array([0.5, 0.3, 0.2])
    b_3 = np.array([0.4, 0.4, 0.2])
    C_3 = np.array([[1.0, 2.0, 4.0],
                    [3.0, 1.0, 2.0],
                    [4.0, 3.0, 1.0]])

    t0 = time.perf_counter()
    P_3, cost_3 = solve_ot(C_3, a_3, b_3)
    elapsed = time.perf_counter() - t0

    src_err, tgt_err = marginal_error(P_3, a_3, b_3)
    print(f"Transport plan P:\n{np.round(P_3, 4)}")
    print(f"Optimal cost:  {cost_3:.4f}  (greedy was 1.1000)")
    print(f"Marginal error — source: {src_err:.2e}  target: {tgt_err:.2e}")
    print(f"Solve time:    {elapsed*1000:.2f} ms")
    assert cost_3 <= 1.1 + 1e-6, "LP should be at least as good as greedy"
    assert src_err < 1e-6 and tgt_err < 1e-6
    print("PASS")

    # ------------------------------------------------------------------
    # Test 3: Trivial identity — one source, one target (cost must be 0
    #         if C is zero, and equal to C[0,0] if not)
    # ------------------------------------------------------------------
    print("\n--- Test 3: 1×1 trivial case ---")
    P_1, cost_1 = solve_ot(np.array([[5.0]]), np.array([1.0]), np.array([1.0]))
    assert abs(cost_1 - 5.0) < 1e-6
    print(f"1×1 cost: {cost_1:.1f}  (expected 5.0)  PASS")

    # ------------------------------------------------------------------
    # Test 4: Uniform marginals — result should be permutation-like
    # ------------------------------------------------------------------
    print("\n--- Test 4: 3×3 identity cost (should recover diagonal plan) ---")
    C_eye = np.array([[0., 1., 1.],
                      [1., 0., 1.],
                      [1., 1., 0.]])
    a_u = np.ones(3) / 3
    b_u = np.ones(3) / 3
    P_eye, cost_eye = solve_ot(C_eye, a_u, b_u)
    print(f"Transport plan:\n{np.round(P_eye, 4)}")
    print(f"Cost: {cost_eye:.4f}  (expected 0.0000)")
    assert abs(cost_eye) < 1e-6
    print("PASS")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
