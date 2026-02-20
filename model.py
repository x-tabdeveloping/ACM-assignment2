from functools import partial

import firetruck as ft
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import plotly.express as px
import plotly.graph_objects as go
from jax.scipy import stats
from plotly.subplots import make_subplots
from tqdm import tqdm

numpyro.set_host_device_count(4)


def take(pytree, index):
    # Indexing a pytree along axis 0
    leaves, treedef = jax.tree.flatten(pytree)
    leaves = [leaf[index] for leaf in leaves]
    return jax.tree.unflatten(treedef, leaves)


def simulate_random(rng_key, rate: float = 0.5, n_trials: int = 120):
    return jax.random.binomial(rng_key, n=1, p=rate, shape=n_trials)


def random_player(rate: float):
    def _choice(rng_key, opponent_choice, prev_choice, params, even: bool = False):
        rate = params["rate"]
        return jax.random.binomial(rng_key, n=1, p=rate), params

    return _choice, {"rate": rate}


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


def plot_bayesian_updates(a, b, n_grid_points=100):
    fig = go.Figure()
    n_trials = a.shape[0]
    colors = px.colors.sample_colorscale("RdBu_r", np.arange(n_trials) / n_trials)
    grid = jnp.linspace(0, 1.0, n_grid_points)
    for alpha, beta, color in zip(a, b, colors):
        density = stats.beta.pdf(grid, a=alpha, b=beta)
        fig.add_scatter(
            x=grid,
            y=density,
            line=dict(color=color),
            showlegend=False,
        )
        fig.add_scatter(
            x=[0.5, 0.5],
            y=[jnp.min(density), jnp.max(density)],
            mode="lines",
            line=dict(dash="dash", color="black"),
            showlegend=False,
        )
    fig.update_layout(template="plotly_white")
    return fig


@ft.compact
def bayesian_rl(self, ys, a=2.0, b=2.0):
    self.lr = dist.Gamma(1, 2)
    self.tau = dist.Gamma(8, 1)
    init_belief = {"a": a, "b": b}
    beliefs = trace_beliefs(init_belief, ys, lr=self.lr)
    self.a = beliefs["a"]
    self.b = beliefs["b"]
    action = apply_concentration(beliefs, tau=self.tau)
    self.a1 = action["a"]
    self.b1 = action["b"]
    return dist.BetaBinomial(self.a1, self.b1, total_count=1)


key = jax.random.key(42)
key, subkey = jax.random.split(key)
ys = simulate_random(subkey, rate=0.9, n_trials=1200)

model = bayesian_rl.add_input(ys)
key, subkey = jax.random.split(key)
simulations = model.sample_predictive(
    subkey, num_samples=10, exclude_deterministic=False
)

for i in range(simulations["a"].shape[0]):
    samples = take(simulations, i)
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        subplot_titles=["Belief distribution", "Action distribution"],
    )
    fig_beliefs = plot_bayesian_updates(samples["a"], samples["b"])
    for trace in fig_beliefs.data:
        fig.add_trace(trace, row=1, col=1)
    fig_actions = plot_bayesian_updates(samples["a1"], samples["b1"])
    for trace in fig_actions.data:
        fig.add_trace(trace, row=1, col=2)
    lr = samples["lr"]
    tau = samples["tau"]
    fig = fig.update_layout(
        template="plotly_white", title=f"lr={lr:.2f}; tau={tau:.2f}"
    )
    fig.show()

sim = take(simulations, 3)
lr = sim["lr"]
tau = sim["tau"]
conditioned = model.condition_on(sim["obs"])
key, subkey = jax.random.split(key)
mcmc = conditioned.sample_posterior(
    subkey,
    num_chains=4,
    num_samples=1000,
    max_tree_depth=10,
    dense_mass=True,
)
print(f"lr={lr:.2f}; tau={tau:.2f}")

posterior = mcmc.get_samples()
px.scatter(x=posterior["lr"], y=posterior["tau"])

ft.plot_trace(mcmc)
ft.plot_forest(mcmc)


model = bayesian_rl.add_input(ys)
key, subkey = jax.random.split(key)
posterior_predictive = model.sample_predictive(
    subkey, posterior_samples=mcmc.get_samples()
)
ft.plot_predictive_check(posterior_predictive, sim["obs"])
