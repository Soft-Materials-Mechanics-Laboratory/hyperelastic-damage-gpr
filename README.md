# Hyperelastic Damage Modeling with Physics-Informed GPR

This repository contains the code and data associated with the manuscript:

**A physics-informed data-driven framework for modeling hyperelastic materials with progressive damage and failure**

---

## Overview

This repository provides the implementation of a physics-informed, data-driven constitutive modeling framework for hyperelastic materials with progressive damage and failure. The framework uses **Gaussian Process Regression (GPR)** in two stages:

1. **Stage I:** learn the constitutive response of the undamaged hyperelastic material
2. **Stage II:** learn the damage evolution behavior and incorporate it into the final constitutive model

The framework is developed and validated using **synthetic data**, and then applied to **experimental brain tissue data** digitized from the literature.

The repository is organized to support transparency and reproducibility of the results presented in the paper.

---

## Repository Structure

```text
hyperelastic-damage-gpr/
│
├── data/           # Synthetic and experimental datasets
├── notebooks/      # Jupyter notebooks reproducing paper results
├── src/            # Reusable Python source files
├── figures/        # Figures used in the README and generated outputs
├── results/        # Saved outputs, exported figures, and model results
├── environment.yml # Conda environment for reproducibility
└── README.md       # Project overview and instructions

