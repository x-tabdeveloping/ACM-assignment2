import firetruck as ft
import jax
import numpyro

from agents import bayesian_rl_agent, simulate_agent
from recovery import simulate_reset_change_rate

numpyro.set_host_device_count(jax.local_device_count())

n_trials = 200
key = jax.random.key(0)
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
subkey, key = jax.random.split(key)
model = bayesian_rl_agent.add_input(ys=ys, rt=rt)
mcmc = model.condition_on(xs).sample_posterior(
    subkey, dense_mass=True, max_tree_depth=12, target_accept_prob=0.9
)

name_mapping = {
    "alpha_rt": "$d_0$",
    "beta_rt": "$\\gamma$",
    "lr": "$\\lambda$",
    "sigma_rt": "$\\sigma$",
}

fig = ft.plot_trace(mcmc)
fig = fig.for_each_annotation(lambda a: a.update(text=name_mapping[a.text]))
fig = fig.update_layout(width=500, height=400)
fig.show()


fig = ft.plot_prior_posterior_update(model, mcmc)
fig = fig.for_each_annotation(lambda a: a.update(text=name_mapping[a.text]))
fig = fig.update_layout(width=500, height=400)
fig.show()


subkey, key = jax.random.split(key)
prior_predictive = bayesian_rl_agent.add_input(ys=ys, rt=rt).sample_predictive(
    subkey, num_samples=4000
)
subkey, key = jax.random.split(key)
posterior_predictive = bayesian_rl_agent.add_input(ys=ys, rt=rt).sample_predictive(
    subkey, posterior_samples=mcmc.get_samples()
)

ft.plot_predictive_check(prior_predictive, obs=xs).update_layout(
    title="Prior Predictive Check", margin=dict(t=50), width=500, height=400
)
ft.plot_predictive_check(posterior_predictive, obs=xs).update_layout(
    title="Posterior Predictive Check", margin=dict(t=50), width=500, height=400
)
