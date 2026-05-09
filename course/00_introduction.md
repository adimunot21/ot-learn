# Chapter 0 — What Is Optimal Transport?

*No formulas in this chapter. Just the idea.*

---

## The Dirt Problem

Imagine you have a pile of dirt and you want to reshape it into a sandcastle. The dirt is in the wrong shape — some of it needs to move a long way, some just needs to shift a little. You want to do this work as efficiently as possible: move exactly the right amount of dirt, exactly as far as needed, no more.

That is Optimal Transport.

The "optimal" part means: find the *cheapest* way to move the dirt. The "transport" part means: move mass (dirt, probability, data) from one distribution to another.

This sounds simple. It turns out to be one of the most powerful ideas in modern mathematics and machine learning.

---

## Why Bother?

Here is a question that sounds easy: **how different are two distributions?**

For example: how different is a distribution of heights in two countries? How different is a face image from a sketch of that face? How different is a cloud of noise samples from a cloud of real data points?

The naive answer is to count mismatches. But this breaks in obvious ways.

**Example:** Suppose distribution A has one point at position 0, and distribution B has one point at position 1. They're different. Now suppose distribution C has one point at position 100. Is A vs C really no different from A vs B?

With simple counting (do the distributions overlap?), both comparisons look equally bad: zero overlap. But intuitively C is *much farther away*. OT captures this — it measures the distance the mass has to travel, so moving mass from 0 to 100 costs far more than from 0 to 1.

This geometry-respecting comparison is exactly what you need for machine learning problems involving distributions.

---

## The Classic Mental Model: Earth Mover's Distance

You have a pile of earth (the source distribution) and a hole of the same total volume (the target distribution). You want to fill the hole by moving earth from the pile.

The **Earth Mover's Distance** is the minimum total work: `Σ (mass moved) × (distance moved)`.

```
Source: pile of earth       Target: hole to fill
   ████                           
   ████    ──────────►      ░░░░░
   ████                     ░░░░░
  (here)                   (there)

Work = (volume of earth) × (distance from pile to hole)
```

If the pile and hole are far apart, the work is high. If they are nearby, the work is low. If they already overlap perfectly, the work is zero.

OT generalises this: the "earth" can be any probability distribution (not just a single pile), and the "hole" can have any shape. The OT problem finds the assignment of source mass to target mass that minimises total work.

---

## The Assignment Problem

Before dealing with continuous distributions, think about a simpler version: the assignment problem.

You have 4 factories and 4 warehouses. Each factory produces goods, each warehouse needs goods. Shipping from factory i to warehouse j costs some amount C[i,j]. You want to ship all goods at minimum total cost.

This is already OT. The "distributions" are the factories and warehouses. The transport plan is the assignment of factory output to warehouse deliveries.

```
Factories         Costs          Warehouses
   F1                 C[1,1]   ──► W1
   F2    ──────────── C[2,3]   ──► W2
   F3                 C[3,2]   ──► W3
   F4                 ...      ──► W4

Find: how much to ship from each Fi to each Wj
Goal: total shipping cost is minimised
```

The larger the problem gets (many factories, many warehouses, continuous distributions), the more you need clever algorithms to solve it. That's what this course builds.

---

## The Coupling Idea

A key concept: instead of a rigid map ("factory 1 goes to warehouse 3"), we allow a **soft assignment**: "factory 1 ships some goods to warehouse 3 and some to warehouse 4."

This soft assignment is called a **transport plan** or **coupling**. It is a table P[i,j] where each entry says: how much mass flows from source i to target j.

The constraints are:
- Each source ships exactly as much as it has (row sums match source)
- Each target receives exactly as much as it needs (column sums match target)
- All entries are non-negative (you can't ship negative goods)

Among all such tables, find the one that minimises total cost. That is the Optimal Transport problem.

---

## Why OT Is Powerful

OT gives you:

1. **A geometry-respecting distance between distributions** — the Wasserstein distance. Unlike other measures, it cares about *where* mass is, not just whether distributions overlap.

2. **A principled way to interpolate between distributions** — if you have a source distribution and a target, OT tells you the "straight-line" path between them. Each intermediate step is a valid distribution.

3. **Better machine learning** — the OT coupling between noise and data turns out to produce dramatically straighter training paths for generative models. This is the secret behind Stable Diffusion 3 and Flux.

---

## What This Course Builds

We start from the assignment problem (Phase 1) and build everything from scratch.

```
Phase 1 (LP)           The exact mathematical formulation of OT
                       ↓
Phase 2 (Duality)      The "price" interpretation — what does each unit of mass cost?
                       ↓
Phase 3 (Sinkhorn)     A fast, practical algorithm to solve OT at scale
                       ↓
Phase 4 (Wasserstein)  OT as a distance metric between distributions
                       ↓
Phase 5 (Barycenters)  Averaging distributions using the OT geometry
                       ↓
Phase 6 (Flow Match)   Training a neural net to generate data via OT paths
```

Every phase uses everything before it. Phase 6 uses the LP (Phase 1) conceptually, the Sinkhorn algorithm (Phase 3) at every training step, and the McCann interpolation (Phase 5) to define the training paths.

---

## Before You Continue

The next chapter (`prerequisites.md`) covers the mathematical background you'll need: what is a probability distribution, what is a linear program, what is KL divergence. Read it if you want those foundations, or skip it and come back when a concept is unclear.

Then start `course/01_discrete_ot.md` — this is where the math begins. The progression will feel natural: every new concept is introduced in terms of the ones you already know.

The code is in `src/`. Every source file is runnable standalone. Running it prints a worked example and cross-validates against an external library.

---

## One Sentence

If you remember one thing from this chapter:

> Optimal Transport is the problem of moving one distribution into another at minimum cost, and the solution is a complete theory of geometry-respecting comparison, interpolation, and generation.
