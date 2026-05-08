# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A deep-dive into Optimal Transport (OT) — mathematics, algorithms, and real applications — built entirely from scratch. Goal: genuine understanding sufficient to derive, implement, and explain every piece without reference. Ends with a novel application to a real domain.

Prior completed projects (same style): Transformer, RL (PPO), AlphaZero, Diffusion models — all from scratch. User is comfortable with PyTorch, numpy, probability, linear algebra, basic optimization. Do not over-explain Python. Do explain OT-specific math from first principles.

## From-Scratch Rules — Non-Negotiable

- Core algorithms implemented from primitives only: numpy, torch, scipy.optimize
- `pot.emd()`, `ot.sinkhorn()`, and equivalents are forbidden until the algorithm already exists in this codebase, written by hand
- Every algorithm must have a worked numerical example with real numbers traced through — not just a formula
- Every tensor/matrix operation must include a shape trace in a comment
- Math comes before code in every phase: derive first, implement second

## Environment

- Python via Miniforge conda, env name: `ot-learn` (`conda activate ot-learn`)
- Core deps: `torch`, `numpy`, `scipy`, `matplotlib`, `pot` (for cross-validation only, after scratch impl)
- GPU: GTX 1650, 4GB VRAM, CUDA 13.0 — flag any experiment that won't fit
- Always run a CPU sanity check before any GPU training

## Project Structure

```
ot-learn/
├── CLAUDE.md
├── PROJECT_PLAN.md
├── src/               # Reusable scratch implementations
├── phases/            # One directory per phase: phase_01/, phase_02/, ...
│   └── phaseNN/
│       ├── derive.md      # Math derivation for this phase (written before code)
│       └── *.py
├── notebooks/         # Plots saved as PNG (matplotlib.use("Agg"))
├── checkpoints/       # Not tracked by git
├── data/              # Not tracked by git
└── course/            # Final multi-chapter course, one .md per phase/topic
```

## Code Conventions (from global CLAUDE.md)

- All hyperparameters in a single `CONFIG` dict at the top with a comment justifying each value
- Every script runnable standalone with `if __name__ == "__main__"` sanity check
- Descriptive names: `cost_matrix` not `C`, `transport_plan` not `T`, `source_weights` not `a`
- Print informative output: OT objective value, iteration count, elapsed time
- Before using any dataset: write an inspection script printing field names, shapes, value ranges, 2–3 raw samples
- Save plots to `notebooks/` as PNG

## Git Workflow

- Init git in Phase 0, commit + push to GitHub after every phase
- `.gitignore`: `__pycache__/`, `*.pyc`, `.DS_Store`, `checkpoints/*.pt`, `data/`, `.ipynb_checkpoints/`, `.env`, `*.log`

## Course Requirements (written after project completes)

Location: `course/`, one markdown file per major phase/topic.
Target reader: knows Python and linear algebra, new to OT.
Each chapter must include:
- The problem being solved and why it matters
- All math with worked numerical examples (trace actual numbers)
- Text-based diagrams for data flow / geometric intuition
- Line-by-line code walkthrough explaining every non-obvious decision
- Shape traces at every step
- Summary table of key concepts
- "What's Next" section
