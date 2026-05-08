# Optimal Transport from Scratch

Deep-dive into Optimal Transport — math, algorithms, and applications — built entirely from primitives.

## Setup

```bash
conda env create -f environment.yml
conda activate ot-learn
```

## Structure

```
src/          reusable scratch implementations
phases/       one directory per phase: derive.md + scripts
notebooks/    saved plots (PNG)
course/       final multi-chapter course
```

## Phases

| Phase | Topic |
|-------|-------|
| 0 | Setup |
| 1 | Discrete OT as a Linear Program |
| 2 | Kantorovich Duality |
| 3 | Sinkhorn Algorithm |
| 4 | Wasserstein Distances |
| 5 | Wasserstein Barycenters |
| 6 | Flow Matching — Generative Models via OT |

See `PROJECT_PLAN.md` for full details.
