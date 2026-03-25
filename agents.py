from functools import partial

import firetruck as ft
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from wald_dist import Wald

numpyro.set_host_device_count(4)


def take(pytree, index):
    # Indexing a pytree along axis 0
    leaves, treedef = jax.tree.flatten(pytree)
    leaves = [leaf[index] for leaf in leaves]
    return jax.tree.unflatten(treedef, leaves)


def update_posterior(state, y, lr=1.0):
    """Updating posterior with the given learning rate,
    which corresponds to the exponent of the likelihood when updating the posterior.
    """
    a = state["a"]
    b = state["b"]
    new_state = {"a": a + lr * y, "b": b + lr * (1 - y)}
    return new_state, new_state


def trace_beliefs(init_state, ys, lr=1.0):
    _, states = jax.lax.scan(partial(update_posterior, lr=lr), init_state, ys)
    return states


def apply_concentration(belief, tau=1.0):
    """Changes the concentration of the b distribution but keeps the mode in-tact
    by manipulating parameters.
    Tau is a multiplier for the concentration parameter of the b distribution.
    """
    a = belief["a"]
    b = belief["b"]
    # Concentration
    k0 = a + b
    # Mode
    w = (a - 1) / (a + b - 2)
    b1 = tau * k0 * (1 - w) + 2 * w - 1
    a1 = (w * b1 - 2 * w + 1) / (1 - w)
    return {"a": a1, "b": b1}


def rt_distribution(belief, alpha, beta, sigma):
    """Returns distribution of reaction times."""
    certainty = belief["a"] + belief["b"]
    drift = alpha + beta * certainty
    return Wald(loc=1 / drift, lam=(1 / sigma) ** 2)


@ft.compact
def bayesian_rl_agent(
    self,
    ys,
    rt=None,
    prior_a=1.0,
    prior_b=1.0,
    alpha_rt_scale=1.0,
    beta_rt_scale=1.0,
    sigma_rt_scale=0.5,
):
    """Bayesian reinforcement-learning agent model in NumPyro."""
    # Lambda/learning rate parameter
    self.lr = dist.Exponential(1.0)
    belief = trace_beliefs({"a": prior_a, "b": prior_b}, ys, lr=self.lr)
    # Priors
    # In the paper I call this d0, it is the intercept of drift
    self.alpha_rt = dist.Exponential(alpha_rt_scale)
    # Gamma param/effect of uncertainty on reaction times
    self.beta_rt = dist.Exponential(beta_rt_scale)
    # Dispersion of the random walk
    self.sigma_rt = dist.Exponential(sigma_rt_scale)
    # Sampling reaction times
    numpyro.sample(
        "rt",
        rt_distribution(belief, self.alpha_rt, self.beta_rt, self.sigma_rt),
        obs=rt,
    )
    return dist.BetaBinomial(belief["a"], belief["b"], total_count=1)


def simulate_agent(
    rng_key,
    ys,
    lr,
    alpha_rt=None,
    beta_rt=None,
    sigma_rt=None,
    prior_a=1.0,
    prior_b=1.0,
):
    """Simulates a run of the agent based on parameters."""
    init_belief = {"a": prior_a, "b": prior_a}
    beliefs = trace_beliefs(init_belief, ys, lr=lr)
    if alpha_rt is None:
        rng_key, subkey = jax.random.split(rng_key)
        alpha_rt = dist.Exponential(1.0).sample(subkey)
    if beta_rt is None:
        rng_key, subkey = jax.random.split(rng_key)
        beta_rt = dist.Exponential(1.0).sample(subkey)
    if sigma_rt is None:
        rng_key, subkey = jax.random.split(rng_key)
        sigma_rt = dist.Exponential(0.5).sample(subkey)
    rng_key, subkey = jax.random.split(rng_key)
    choices = dist.BetaBinomial(beliefs["a"], beliefs["b"], total_count=1).sample(
        subkey
    )
    rng_key, subkey = jax.random.split(rng_key)
    rt_dist = rt_distribution(beliefs, alpha_rt, beta_rt, sigma_rt)
    rt = rt_dist.sample(subkey)
    params = {"alpha_rt": alpha_rt, "beta_rt": beta_rt, "lr": lr, "sigma_rt": sigma_rt}
    return choices, rt, params
