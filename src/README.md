# Source Code Overview

This folder contains reusable Python modules implementing the core components of the constitutive modeling framework:

- Stage I GPR (undamaged hyperelastic response)
- Stage II GPR (damage evolution)
- Constitutive model evaluation

---

## Synthetic vs Experimental Implementations

Two sets of modules are provided to reflect a key modeling distinction between synthetic and experimental datasets.

### Synthetic data

- The material response includes **both volumetric and isochoric contributions**
- Stage I learns:
  - volumetric response function (\zeta₁)
  - isochoric response functions (\Gamma₁, \Gamma₂)
- No incompressibility constraint is imposed

---

### Experimental data

- The material response is assumed **incompressible (J = 1)**
- The volumetric response is **not learned explicitly**
- A **Lagrange multiplier enforces incompressibility** through boundary conditions
- Stage I is modified to learn **only the isochoric response functions (\Gamma₁, \Gamma₂)**

This leads to separate implementations of:

- kinematics (incompressibility constraint)
- constitutive modeling (modified Stage I formulation)

Details of this formulation are provided in the manuscript.

---

## Module Structure

- `synthetic_*` modules:
  - General compressible formulation
  - Used for synthetic data validation

- `experimental_*` modules:
  - Incompressible formulation (J = 1)
  - Used for experimental data analysis

- `constrained_gpr.py`:
  - Shared between both cases
  - Implements Stage II constrained GPR for damage evolution

---

## Notes

- All implementations follow the same overall two-stage framework
- Differences arise only in Stage I due to physical modeling assumptions
- The constrained GPR formulation (Stage II) is identical across both cases
