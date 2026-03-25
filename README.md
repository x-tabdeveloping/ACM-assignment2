# ACM-assignment2
Second assignment for Advanced Cognitive Modelling.
In this assignment I implement a Bayesian Reinforcement Learning agent and try to do parameter recovery with MCMC.

The structure of the repo is as such:

 - `agents.py` contains code to simulate the agent, and the NumPyro model for parameter recovery.
 - `wald_dist.py` contains a custom implementation of the Wald distribution, since it is not implemented in NumPyro yet.
 - `recovery.py` runs the model recovery experiment with varying gamma, and outputs results in the `results/` folder.
 - `trials.py`  runs model recovery simulations with varying N trials, and outputs result in the `results/n_trials/` directory .
 - `plot_recovery.py` plots the results of model recovery (Figures 3 and 4).
 - `model_checks.py` plots and computes model checks on a simulated dataset (Figure 1 and Table 1).
 - `prior_sensitivity.py` plots the posterior under a power-scaled prior (Figure 2).
