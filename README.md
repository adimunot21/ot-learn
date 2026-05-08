# Optimal Transport from Scratch

A complete, from-scratch implementation of Optimal Transport theory — every algorithm
derived by hand before being coded, every result cross-validated. Ends with
OT-coupled flow matching, the generative modelling framework used in Stable Diffusion 3 and Flux.

**Stack:** Python · NumPy · SciPy · PyTorch · no OT libraries until cross-validation

## Setup

```bash
conda env create -f environment.yml
conda activate ot-learn
```

## What Was Built

| Phase | Topic | Key result |
|-------|-------|-----------|
| 1 | Discrete OT as a Linear Program | Exact solver matches `pot.emd()` to 1.78e-15 |
| 2 | Kantorovich Duality | Strong duality gap = 0 on 50 random trials |
| 3 | Sinkhorn Algorithm | Log-domain + vanilla; matches `pot.sinkhorn()` to 2.5e-9 |
| 4 | Wasserstein Distances | 1D quantile, Gaussian Bures, Sinkhorn W₂, sliced |
| 5 | Wasserstein Barycenters | Circle+square barycenter equidistant to both (asymmetry 0.018) |
| 6 | Flow Matching with OT Coupling | OT paths **61–140× straighter** than independent coupling |

## Source Files

```
src/
  lp_ot.py          discrete OT via scipy LP (Phase 1)
  dual_ot.py        Kantorovich dual, c-transform, complementary slackness (Phase 2)
  sinkhorn.py       vanilla + log-domain Sinkhorn (Phase 3)
  wasserstein.py    W₁/W₂ 1D, Gaussian Bures, Sinkhorn W₂, sliced (Phase 4)
  barycenter.py     McCann interpolation, free-support barycenter (Phase 5)
  flow_matching.py  VelocityMLP, mini-batch OT coupling, Euler ODE (Phase 6)
```

Each file is runnable standalone (`python src/<file>.py`) and includes sanity checks.

## The Thread

Every phase feeds the next:

```
Phase 1 (LP)         defines "optimal" transport
Phase 2 (Duality)    dual potentials = initial velocity of the flow
Phase 3 (Sinkhorn)   makes OT fast enough for a training loop
Phase 4 (Wasserstein) OT cost as a geometry-respecting distance
Phase 5 (McCann)     z_t = (1-t)x₀ + tx₁  is the W₂-geodesic
Phase 6 (Flow Match) regress a neural net onto those geodesic velocities
```

## Course

`course/` contains six Markdown chapters — written after each phase, while the
derivations were fresh. Each chapter includes full math with worked numerical
examples, text diagrams, line-by-line code walkthrough, and shape traces.

Target reader: knows Python and linear algebra, new to OT.

## Honest Limitations

Three real constraints discovered during the project:

**1. Sinkhorn diverges for very small ε (Phase 3)**
Log-domain Sinkhorn stays numerically finite for any ε, but convergence requires
~1/ε × (iterations needed at ε=0.1) steps. At ε=1e-4 this is millions of steps —
impractical. For ε < 0.01, convergence is slow even in log-domain. In practice
ε = 0.05–0.1 is the useful range for most applications.

**2. Sliced Wasserstein underestimates W₂ by 1/√d (Phase 4)**
Sliced W₂ averages 1D projections, so for unit-covariance Gaussians shifted by μ
in ℝᵈ, `SW₂ ≈ W₂ / √d`. In 2D this is a 30% underestimate. Sliced W₂ is not
a bad approximation of W₂ — it answers a slightly different question and is the
only tractable option in high dimensions. The correct reference value is W₂/√d,
not W₂.

**3. Barycenter with k=1 doesn't recover the input exactly (Phase 5)**
With a single reference measure, the Wasserstein barycenter should return it
unchanged. Our implementation gets W₂ ≈ 0.2 instead of ~0, due to random
initialisation and Sinkhorn regularisation blurring. Tighter with smaller ε or
smarter initialisation. Not a problem for k≥2 use cases.

## Key Numbers

```
Phase 1: LP vs pot.emd()            max cost diff = 1.78e-15  (machine precision)
Phase 2: duality gap (50 trials)    max gap       = 1.78e-15
Phase 3: Sinkhorn vs pot.sinkhorn() max cost diff = 2.52e-09
Phase 4: W₂ Gaussian closed form    exact to 6 decimal places
Phase 5: barycenter asymmetry       0.018  (equal λ → expected 0)
Phase 6: OT vs independent loss     0.048 vs 1.333  (moons)
Phase 6: path straightness          OT is 61× (moons) / 140× (8-gaussians) straighter
```
