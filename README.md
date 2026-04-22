# AIMTG-VEHICLE-Core

Numerical experiments for the VEHICLE framework — a relational dynamical system defined on component-structured graphs with internal coupling constraints.

---

## Overview

The VEHICLE model describes a discrete dynamical system driven by the minimization of a global tension functional combining:

- inter-node discrepancy terms (external consistency)
- intra-node coherence constraints (internal structural coupling)

This formulation induces a non-trivial interaction between local alignment and global compatibility, distinguishing the system from classical consensus dynamics.

---

## Objectives

The experiments in this repository aim to validate:

- monotonic dissipation of the global tension functional  
- emergence of structural coherence under internal coupling  
- qualitative deviation from standard consensus behavior  
- stability and convergence patterns of the dynamical system  

---

## Experiments

### Experiment 1 — Base Dynamics

Validation of global tension dissipation and convergence behavior under the baseline configuration.

---

### Experiment 2 — Coupling Comparison

Comparison between:

- λ = 0 (uncoupled system, consensus-like behavior)
- λ > 0 (internally coupled system)

Key result:
Internal coupling induces structural coherence and prevents trivial consensus.

---

### Experiment 3 — Lambda Sweep

Systematic analysis of the effect of the coupling parameter λ.

Key observations:

- increasing λ enhances internal coherence  
- introduces structural constraints  
- modifies the equilibrium configuration  

---

## Requirements

- Python 3.x  
- numpy  
- matplotlib  

---

## How to Run

Example:

```bash
python experiments/exp3_lambda_sweep.py
