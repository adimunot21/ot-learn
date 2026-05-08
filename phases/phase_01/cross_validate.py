"""
Cross-validate src/lp_ot.py against pot.emd().

Run this AFTER the scratch implementation exists (which it does).
The point is not to use POT for OT — it's to confirm our LP solver
produces the same transport cost as the reference implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import ot  # POT library — only used here for cross-validation

from src.lp_ot import solve_ot, transport_cost

CONFIG = {
    "n_random_trials": 20,
    "rng_seed": 42,
    "cost_tol": 1e-5,   # tolerance for cost agreement between scratch vs POT
}


def random_ot_instance(rng: np.random.Generator, n: int, m: int):
    a = rng.dirichlet(np.ones(n))  # random probability vector, shape (n,)
    b = rng.dirichlet(np.ones(m))  # shape (m,)
    C = rng.uniform(0, 10, size=(n, m))  # shape (n, m)
    return a, b, C


def cross_validate():
    rng = np.random.default_rng(CONFIG["rng_seed"])
    print("Cross-validating scratch LP solver against pot.emd()")
    print(f"Running {CONFIG['n_random_trials']} random trials...\n")

    max_cost_diff = 0.0

    for trial in range(CONFIG["n_random_trials"]):
        n = rng.integers(2, 8)
        m = rng.integers(2, 8)
        a, b, C = random_ot_instance(rng, int(n), int(m))

        # Scratch solver
        P_scratch, cost_scratch = solve_ot(C, a, b)

        # POT reference
        P_pot = ot.emd(a, b, C)
        cost_pot = transport_cost(C, P_pot)

        diff = abs(cost_scratch - cost_pot)
        max_cost_diff = max(max_cost_diff, diff)
        status = "OK" if diff < CONFIG["cost_tol"] else "FAIL"
        print(f"  Trial {trial+1:2d} | n={n} m={m} | "
              f"scratch={cost_scratch:.6f}  pot={cost_pot:.6f}  diff={diff:.2e}  {status}")

        if status == "FAIL":
            print(f"    MISMATCH EXCEEDS TOLERANCE {CONFIG['cost_tol']}")
            sys.exit(1)

    print(f"\nMax cost difference across all trials: {max_cost_diff:.2e}")
    print(f"All {CONFIG['n_random_trials']} trials within tolerance {CONFIG['cost_tol']}.")
    print("\nScratch LP solver matches pot.emd() — implementation validated.")


if __name__ == "__main__":
    cross_validate()
