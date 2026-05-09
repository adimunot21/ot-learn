# Optimal Transport — A Complete Course

**From intuition to generative models. Every concept built from scratch.**

This is a self-contained course that teaches Optimal Transport (OT) to someone who knows Python and basic linear algebra. No prior knowledge of OT, probability theory, or machine learning is assumed. Every algorithm is derived before it is coded. Every result is cross-validated.

The course ends with **Flow Matching** — the generative modelling framework powering Stable Diffusion 3, Flux, and Sora. All the OT theory you build along the way is exactly what makes those models work.

---

## Who This Is For

You will get the most out of this if you:
- Know Python (you can read a for-loop and a NumPy array)
- Have seen a matrix before
- Are curious about either generative AI or mathematical optimisation

You do **not** need:
- A background in machine learning
- Prior exposure to probability theory or measure theory
- Any knowledge of Optimal Transport

---

## How to Use This Repo

The repo has three layers, all teaching the same ideas at different depths.

```
course/          ← Start here. Plain English + math + diagrams + worked examples.
phases/          ← Deeper. Full derivations with every step shown.
src/             ← Code. Implementations you can run and inspect.
```

**Recommended path:**

1. Read `course/00_introduction.md` — intuition, no formulas, understand the big picture
2. Read `course/prerequisites.md` if you want a math refresher (or skip if confident)
3. For each phase (1–6): read the chapter → read the derivation → run the code
4. Keep `course/glossary.md` open in a second tab while reading

**If you just want to run things:**

```bash
conda env create -f environment.yml
conda activate ot-learn
python src/lp_ot.py          # Phase 1 — LP solver + cross-validation
python src/dual_ot.py        # Phase 2 — Kantorovich duality
python src/sinkhorn.py       # Phase 3 — Sinkhorn algorithm
python src/wasserstein.py    # Phase 4 — Wasserstein distances
python src/barycenter.py     # Phase 5 — Wasserstein barycenters
python src/flow_matching.py  # Phase 6 — Flow matching (trains a neural net!)
```

Each file prints its results and saves plots to `notebooks/`.

---

## The Six Phases

| # | What You'll Learn | Why It Matters |
|---|---|---|
| 0 | What OT is, without any formulas | Mental model for everything that follows |
| 1 | OT as a linear program | The exact, principled way to move mass |
| 2 | Kantorovich duality | Prices that reveal what transport "costs" |
| 3 | Sinkhorn algorithm | Makes OT fast enough to use in practice |
| 4 | Wasserstein distances | OT as a geometry-respecting metric |
| 5 | Wasserstein barycenters | Averaging distributions without blurring |
| 6 | Flow Matching | Training generative models using OT paths |

Each phase builds on the previous one. By Phase 6 you will have used every concept from Phases 1–5.

---

## Course Files

```
course/
  00_introduction.md     What is OT? (zero formulas, all intuition)
  prerequisites.md       Math primer: distributions, LPs, KL divergence
  glossary.md            Plain-English OT dictionary
  01_discrete_ot.md      LP formulation, transport plan, marginals
  02_duality.md          Dual problem, c-transform, complementary slackness
  03_sinkhorn.md         Entropy regularisation, Sinkhorn iterations, log-domain
  04_wasserstein.md      W₁/W₂ distances, Gaussian closed form, sliced
  05_barycenters.md      McCann interpolation, fixed-point barycenter algorithm
  06_flow_matching.md    Conditional FM objective, OT coupling, Euler ODE
```

---

## Key Results

```
Phase 1: our LP solver vs pot.emd()          max cost diff = 1.78e-15  (machine precision)
Phase 2: strong duality gap (50 trials)      max gap       = 1.78e-15
Phase 3: Sinkhorn vs pot.sinkhorn()          max cost diff = 2.52e-09
Phase 4: W₂ Gaussian closed form            exact to 6 decimal places
Phase 5: barycenter equidistance (k=2)       asymmetry = 0.018  (expected: 0)
Phase 6: OT vs independent coupling         OT is 61× (moons) / 140× (8-gaussians) straighter
Phase 6: training loss comparison           OT: 0.048,  independent: 1.333  (moons)
```

---

## Honest Limitations

Three genuine constraints discovered during the project:

**1. Sinkhorn is slow for very small ε (Phase 3)**
Log-domain Sinkhorn stays numerically stable for any ε, but convergence slows down as ~1/ε. At ε=1e-4 you'd need millions of iterations. The practical range is ε = 0.05–0.1.

**2. Sliced Wasserstein underestimates W₂ by 1/√d (Phase 4)**
Sliced W₂ averages 1D projections. For d-dimensional Gaussians this gives `SW₂ ≈ W₂/√d`. In 2D that's a 30% underestimate. This is correct behaviour — sliced W₂ answers a slightly different (but useful) question.

**3. Barycenter with k=1 doesn't recover the input exactly (Phase 5)**
With one reference measure, the barycenter should return it unchanged. We get W₂ ≈ 0.2 instead of ~0, due to random initialisation and Sinkhorn regularisation blurring. Not a problem for k≥2.

---

## Stack

Python · NumPy · SciPy · PyTorch · no OT libraries until cross-validation

The rule throughout: implement from scratch first, then cross-validate against `POT` (Python Optimal Transport).
