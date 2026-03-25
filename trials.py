from pathlib import Path

import jax
import jax.numpy as jnp
import joblib
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from tqdm import tqdm

from agents import bayesian_rl_agent, simulate_agent
from recovery import simulate_reset_change_rate

numpyro.set_host_device_count(4)

out_dir = Path("results/n_trials")
out_dir.mkdir(exist_ok=True, parents="true")
key = jax.random.key(0)
beta_rt = 1.0
lrs = jnp.linspace(0, 2.0, 5)
for i, n_trials in enumerate([150, 250, 500, 750, 1000, 2000]):
    for j, lr in enumerate(lrs):
        print(f"Running beta_rt={beta_rt}; lr={lr}")
        key, subkey = jax.random.split(key)
        ys = simulate_reset_change_rate(
            subkey,
            learning_phase=50,
            unlearning_phase=50,
            rate_shape=0.1,
            n_trials=n_trials,
        )
        key, subkey = jax.random.split(key)
        xs, rt, params = simulate_agent(
            subkey,
            ys,
            lr=lr,
            beta_rt=beta_rt,
            alpha_rt=0.1,
            sigma_rt=0.2,
            prior_a=1.0,
            prior_b=1.0,
        )
        model = bayesian_rl_agent.add_input(ys, rt=rt).condition_on(xs)
        key, subkey = jax.random.split(key)
        mcmc = model.sample_posterior(subkey, max_tree_depth=12, dense_mass=True)
        print(params)
        samples = mcmc.get_samples(group_by_chain=True)
        extra_fields = mcmc.get_extra_fields(group_by_chain=True)
        _summary = summary(samples, group_by_chain=True)
        res = {
            "params": params,
            "samples": samples,
            "summary": _summary,
            "extra_fields": extra_fields,
        }
        print("Saving run...")
        joblib.dump(res, out_dir.joinpath(f"recovery_{i}_{j}.joblib"))
