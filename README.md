# Angular-Momentum

Code and reproducible analysis pipeline accompanying the manuscript:

**"Gaia DR3 Open Clusters Do Not Require a Universal Early Angular-Momentum Transition"**

## Overview

This repository contains the analysis scripts used to investigate whether Galactic open clusters exhibit evidence for a universal early angular-momentum transition or a characteristic $\sim70$ Myr despinning clock.

Using Gaia DR3 kinematic data and the cleaned open-cluster catalog of Hunt & Reffert (2024), we measure a normalized angular-momentum proxy


f_L = \frac{L_{\rm obs}}{L_{\rm vir}},


and test several population-level evolutionary models:

* Single power-law evolution
* Broken power-law evolution
* Two-phase empirical despinning model

The primary conclusion of the study is that, after accounting for intrinsic cluster-to-cluster scatter, current Gaia DR3 data do **not require** a universal early angular-momentum transition.



## Repository Structure


Angular-Momentum/
│
├── fossil_angular_momentum.py
├── scheme_v3_analysis.py
├── robustness_checks.py
├── explain_70myr_analysis.py
└── README.md


### fossil_angular_momentum.py

Main pipeline for deriving cluster angular-momentum measurements.

Functions include:

* Gaia DR3 cluster member processing
* Velocity-gradient tensor fitting
* Angular velocity estimation
* Calculation of


L_{\rm obs}


and


f_L


for each cluster.

Output:

* cluster-level angular-momentum catalog
* velocity-gradient diagnostics
* quality metrics



### scheme_v3_analysis.py

Population-level statistical analysis.

Implements:

* Single power-law model
* Broken power-law model
* Bayesian broken-power-law inference
* Intrinsic-scatter likelihood modeling
* BIC model comparison

Output:

* best-fit parameters
* transition-age estimates
* Bayesian posterior samples
* model-comparison statistics



### robustness_checks.py

Robustness and systematic-error evaluation.

Tests include:

* Bootstrap resampling
* Jackknife analysis
* Age-uncertainty propagation
* Membership-threshold variations
* RV-number threshold tests
* Sample-selection sensitivity

Output:

* robustness tables
* bootstrap distributions
* systematic-error budget



### explain_70myr_analysis.py

Diagnostic investigation of the apparent ~70 Myr transition.

This script evaluates:

* relaxation-time interpretation
* mass dependence
* Galactocentric-radius dependence
* environmental effects
* selection-function sensitivity

Output:

* break-age diagnostics
* environmental trend figures
* relaxation-time comparisons



## Data

The analysis is based on:

### Gaia DR3

Gaia Collaboration et al. (2023)

### Open-cluster catalog

Hunt & Reffert (2024)

*Improving the Open Cluster Census III:
Using Cluster Masses, Radii, and Dynamics to Create a Cleaned Open Cluster Catalogue*

A&A, 686, A42

Users must obtain the original catalog separately and place it in the expected data directory.


## Reproducing the Analysis

### Step 1

Generate cluster angular-momentum measurements:

 
python fossil_angular_momentum.py
 

### Step 2

Run population-model fitting:

 
python scheme_v3_analysis.py
 

### Step 3

Perform robustness tests:

 
python robustness_checks.py
 

### Step 4

Investigate the apparent 70 Myr feature:

 
python explain_70myr_analysis.py
 

## Main Scientific Result

A broken power-law fit can produce a conditional transition age near

 
t_b \sim 70\ {\rm Myr},
 

when only formal measurement uncertainties are considered.

However, after introducing intrinsic cluster-to-cluster scatter,

 
\sigma_{\rm int},
 

the single power-law model becomes statistically preferred.

Consequently:

> Current Gaia DR3 open-cluster data do not require a universal early angular-momentum transition or a universal ~70 Myr despinning clock.

Instead, the observed diversity is consistent with substantial cluster-to-cluster variation in dynamical histories.

 

## Citation

If you use this code, please cite:

Chi, H. (2026)

*Gaia DR3 Open Clusters Do Not Require a Universal Early Angular-Momentum Transition*

(submitted)

 

## License

This repository is released for academic and research use.

Please cite the associated publication when using the code or derived products.

 

## Contact

Huanbin Chi

School of Artificial Intelligence
Yunnan Open University, China

Center for Astrophysics
Guangzhou University, China

Email: [chihuanbin@126.com] 
