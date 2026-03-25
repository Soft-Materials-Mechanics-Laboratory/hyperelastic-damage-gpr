# Hyperelastic Damage and Failure Modeling with Physics-Informed GPR

This repository contains the code and data associated with the manuscript:

**Upadhyay, K. (2026). A physics-informed data-driven framework for modeling hyperelastic materials with progressive damage and failure. arXiv Preprint arXiv:2602.11414.**

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
```

---

## Contents

This repository will include:

- Jupyter notebooks used to generate results for:
  - synthetic data validation  
  - application to experimental brain tissue datasets  
- associated input datasets  
- reusable Python source code extracted from the notebooks  
- environment specification for reproducibility  
- figures and outputs relevant to the manuscript  

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Soft-Materials-Mechanics-Laboratory/hyperelastic-damage-gpr.git
cd hyperelastic-damage-gpr
````

### 2. Create the Python environment

```bash
conda env create -f environment.yml
conda activate hyperelastic-damage-gpr
````

### 3. Launch Jupyter
````bash
jupyter lab
````

### 4. Run the notebooks
The notebooks in the notebooks/ folder are intended to reproduce the main results of the manuscript.

A recommended workflow is:

1. Run the synthetic-data notebook(s) first
2. Run the experimental-data notebook(s) next
3. Compare generated figures and outputs with those reported in the manuscript
