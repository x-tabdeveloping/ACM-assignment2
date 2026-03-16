from functools import partial

import firetruck as ft
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

numpyro.set_host_device_count(4)


def take(pytree, index):
    # Indexing a pytree along axis 0
    leaves, treedef = jax.tree.flatten(pytree)
    leaves = [leaf[index] for leaf in leaves]
    return jax.tree.unflatten(treedef, leaves)


def simulate_random(rng_key, rate: float = 0.5, n_trials: int = 120):
    return jax.random.binomial(rng_key, n=1, p=rate, shape=n_trials)


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


def rt_distribution(alpha, beta, uncertainty):
    return dist.LogNormal(beta * uncertainty, alpha)


@ft.compact
def bayesian_rl_agent(self, ys, rt=None, prior_a=1.0, prior_b=1.0):
    self.lr = dist.Exponential(1.0)
    belief = trace_beliefs({"a": prior_a, "b": prior_b}, ys, lr=self.lr)
    if rt is not None:
        self.alpha_rt = dist.Exponential(1.0)
        self.beta_rt = dist.Normal(1.0, 1.0)
        uncertainty = 1 / (belief["a"] + belief["b"])
        numpyro.sample(
            "rt",
            rt_distribution(self.alpha_rt, self.beta_rt, uncertainty),
            obs=rt,
        )
    return dist.BetaBinomial(belief["a"], belief["b"], total_count=1)


def simulate_agent(
    rng_key, ys, lr, alpha_rt=None, beta_rt=None, prior_a=1.0, prior_b=1.0
):
    init_belief = {"a": prior_a, "b": prior_a}
    beliefs = trace_beliefs(init_belief, ys, lr=lr)
    if alpha_rt is None:
        rng_key, subkey = jax.random.split(rng_key)
        alpha_rt = dist.Exponential(1.0).sample(subkey)
    if beta_rt is None:
        rng_key, subkey = jax.random.split(rng_key)
        beta_rt = dist.Normal(0, 0.5).sample(subkey)
    rng_key, subkey = jax.random.split(rng_key)
    choices = dist.BetaBinomial(beliefs["a"], beliefs["b"], total_count=1).sample(
        subkey
    )
    uncertainty = 1 / (beliefs["a"] + beliefs["b"])
    rng_key, subkey = jax.random.split(rng_key)
    rt = rt_distribution(alpha_rt, beta_rt, uncertainty).sample(subkey)
    return choices, rt
