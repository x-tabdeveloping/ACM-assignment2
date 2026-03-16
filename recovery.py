from pathlib import Path

import jax
import jax.numpy as jnp
import joblib
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from tqdm import tqdm

from agents import bayesian_rl_agent, simulate_agent


def simulate_reset_change_rate(
    rng_key, learning_phase=50, unlearning_phase=50, rate_shape=0.5, n_trials=150
):
    """Simulates random opponent that changes the rate of heads every N steps,
    and has M "unlearning" steps between runs, with 0.5 head rate,
    so that the agent's beliefs are reset."""
    ys = []
    while len(ys) < n_trials:
        rng_key, subkey = jax.random.split(rng_key)
        rate_next = dist.Beta(rate_shape, rate_shape).sample(subkey)
        rng_key, subkey = jax.random.split(rng_key)
        _y = jax.random.binomial(subkey, n=1, p=rate_next, shape=int(learning_phase))
        ys.extend(_y)
        rng_key, subkey = jax.random.split(rng_key)
        _y = jax.random.binomial(subkey, n=1, p=0.5, shape=int(unlearning_phase))
        ys.extend(_y)
    return jnp.array(ys[:n_trials])


def main():
    out_dir = Path("results/")
    out_dir.mkdir(exist_ok=True)
    key = jax.random.key(0)
    beta_rts = jnp.linspace(0.1, 2.0, 5)
    lrs = jnp.linspace(0, 2.0, 5)
    for i, beta_rt in enumerate(tqdm(beta_rts, desc="Running all simulations")):
        for j, lr in enumerate(lrs):
            print(f"Running beta_rt={beta_rt}; lr={lr}")
            key, subkey = jax.random.split(key)
            ys = simulate_reset_change_rate(
                subkey,
                learning_phase=50,
                unlearning_phase=50,
                rate_shape=0.1,
                n_trials=1000,
            )
            key, subkey = jax.random.split(key)
            xs, rt, params = simulate_agent(
                subkey,
                ys,
                lr=lr,
                beta_rt=beta_rt,
                alpha_rt=0.2,
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


if __name__ == "__main__":
    main()
