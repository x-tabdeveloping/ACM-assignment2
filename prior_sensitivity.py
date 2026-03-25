from pathlib import Path

import jax
import jax.numpy as jnp
import numpyro
import plotly.express as px
import plotly.graph_objects as go
from jax.scipy.stats import gaussian_kde
from plotly.subplots import make_subplots

from agents import bayesian_rl_agent, simulate_agent
from recovery import simulate_reset_change_rate

numpyro.set_host_device_count(jax.local_device_count())

n_trials = 200
key = jax.random.key(0)
scales = [0.5, 0.8, 1.0, 1.2, 1.5]
posteriors = []
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
    lr=0.2,
    beta_rt=0.5,
    alpha_rt=0.1,
    sigma_rt=0.2,
    prior_a=1.0,
    prior_b=1.0,
)
for scale in scales:
    subkey, key = jax.random.split(key)
    model = bayesian_rl_agent.add_input(
        ys=ys,
        rt=rt,
        lr_scale=1.0 * scale,
        alpha_rt_scale=1.0 * scale,
        beta_rt_scale=1.0 * scale,
        sigma_rt_scale=0.5 * scale,
    )
    mcmc = model.condition_on(xs).sample_posterior(
        subkey, dense_mass=True, max_tree_depth=12, target_accept_prob=0.9
    )
    posteriors.append(mcmc.get_samples())

subkey, key = jax.random.split(key)
model = bayesian_rl_agent.add_input(
    ys=ys,
    rt=rt,
    lr_scale=10,
    alpha_rt_scale=1.0,
    beta_rt_scale=1.0,
    sigma_rt_scale=0.5,
)
mcmc = model.condition_on(xs).sample_posterior(
    subkey, dense_mass=True, max_tree_depth=12, target_accept_prob=0.9
)

name_mapping = {
    "alpha_rt": "$d_0$",
    "beta_rt": "$\\gamma$",
    "lr": "$\\lambda$",
    "sigma_rt": "$\\sigma$",
}
var_order = list(name_mapping.keys())
fig = make_subplots(
    rows=1,
    cols=len(var_order),
    subplot_titles=[name_mapping[var_id] for var_id in var_order],
)
colors = px.colors.qualitative.Bold
for i_scale, (scale, posterior) in enumerate(zip(scales, posteriors)):
    for i_var, var_id in enumerate(var_order):
        samples = jnp.ravel(posterior[var_id])
        dens = gaussian_kde(samples)
        grid = jnp.linspace(samples.min(), samples.max(), 200)
        fig.add_scattergl(
            x=grid,
            y=dens.pdf(grid),
            line=dict(color=colors[i_scale]),
            name=f"$\\alpha={scale}$",
            showlegend=i_var == 0,
            row=1,
            col=i_var + 1,
        )
fig = fig.update_layout(
    template="plotly_white",
    margin=dict(l=5, r=5, b=5, t=20),
    width=1100,
    height=300,
    font=dict(size=16),
)
fig.show()

figs_dir = Path("figures")
figs_dir.mkdir(exist_ok=True)
fig.write_image(
    figs_dir.joinpath("prior_sensitivity.png"),
    scale=2,
    width=1100,
    height=300,
)
