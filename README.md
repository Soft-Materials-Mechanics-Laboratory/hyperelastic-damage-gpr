# Hyperelastic Damage and Failure Modeling with Physics-Informed GPR

This repository contains the code and data associated with the manuscript:

**Upadhyay, K. (2026). A physics-informed data-driven framework for modeling hyperelastic materials with progressive damage and failure. arXiv Preprint arXiv:2602.11414.** 
<https://doi.org/10.48550/arXiv.2602.11414>

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
The notebooks in the `notebooks/` folder are intended to reproduce the main results of the manuscript.

A recommended workflow is:

1. Run the synthetic-data notebook(s) first
2. Run the experimental-data notebook(s) next
3. Compare generated figures and outputs with those reported in the manuscript

---

## Data

This repository contains two categories of data:
- Synthetic Data: Synthetic datasets are generated and/or used for model development, validation, and demonstration of the framework.
- Experimental Data: Experimental brain tissue datasets included here were digitized from published literature using WebPlotDigitizer for the purpose of reproducing the analyses in this study.

Please cite the original experimental study when using these data:

G. Franceschini, D. Bigoni, P. Regitnig, G. A. Holzapfel, Brain tissue deforms similarly to filled elastomers and follows consolidation theory, Journal of the Mechanics and Physics of Solids 54 (12) (2006) 2592–2620. doi:10.1016/j.jmps.2006.05.004.

## Reproducibility

The goal of this repository is to make the main results of the paper transparent and reproducible.

To improve reproducibility:
- All required Python dependencies are listed in `environment.yml`
- Notebooks are organized to follow the workflow used in the paper
- Reusable code will be placed in the `src/` directory
- Input data and generated outputs are separated into dedicated folders

Please note that minor numerical differences may occur depending on:
- Package versions
- Operating system
- Random initialization (where applicable)

---

## Method Summary

The constitutive modeling framework in this work combines physical structure and machine learning.

At a high level:
- The undamaged hyperelastic response is learned first
- The damage behavior is then learned separately in a second stage
- The final constitutive model combines these two learned components
- The approach is designed to remain physically meaningful while retaining flexibility from data-driven modeling

This structure allows the framework to capture progressive damage and failure while preserving interpretability.

---

## Recommended Citation

Please cite this research as:

````bash
@article{Upadhyay2026,
archivePrefix = {arXiv},
arxivId = {2602.11414},
author = {Upadhyay, Kshitiz},
eprint = {2602.11414},
keywords = {brain tissue mechanics,damage and failure,data-driven constitutive models,gaussian process regression,hyperelasticity,machine learning,physics-informed,soft materials},
title = {{A physics-informed data-driven framework for modeling hyperelastic materials with progressive damage and failure}},
url = {http://arxiv.org/abs/2602.11414},
year = {2026}
}
````

---

## License

This repository is released under the MIT License.

See the `LICENSE` file for details.

---

## Author

**Kshitiz Upadhyay, Ph.D.**
Assistant Professor
Director of the Soft Materials Mechanics Laboratory
Department of Aerospace Engineering and Mechanics
University of Minnesota
Email: kshitizu@umn.edu

---

## Acknowledgment

This research is based upon work supported by the National Science Foundation under Grant No. 2331294. This repository was created to support open and reproducible research in constitutive modeling, soft material mechanics, and scientific machine learning.
