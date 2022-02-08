# BAGEL
BAGEL: Bayesian Analysis of Gene Expression Lineages

BAGEL can be run by opening 1. Algorithm directory and running the Algorithm.py file. This will produce BAGELS mouse bone marrow dataset results.

# Model parameters can be set in:
## 1.) Algorithm/Algorithm.py
-
-
-
## 1.) Algorithm/lineage_parameters
### Window parameters                                     ________
-step_size = 200  # How much cells per step (Step size)  |_______/
-window_size = 150  # How many cells in window  (Widow size) |_____|

### Gibbs sampler parameters
-itterations = 2000  # Total itterations
-burn_period = 500  # Total itterations to burn "burn period"

