import jax
import jax.numpy as jnp
from jax import lax
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key


def logpow(x, m):
    """Translated to JAX from PyMC"""
    return jnp.where(x == 0, jnp.where(m == 0, 0.0, -jnp.inf), m * jnp.log(x))


class Wald(Distribution):
    """Inverse Normal distribution in NumPyro"""

    arg_constraints = {
        "loc": constraints.positive,
        "shape": constraints.positive,
    }
    reparametrized_params = ["loc", "shape"]
    support = constraints.positive

    def __init__(self, loc=1.0, lam=1.0, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(lam))
        self.loc, self.lam = promote_shapes(loc, lam, shape=batch_shape)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """From wikipedia"""
        out_shape = sample_shape + self.batch_shape + self.event_shape
        assert is_prng_key(key)
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, shape=out_shape)
        y = v * v
        x = (
            self.loc
            + (self.loc * self.loc * y) / (2 * self.lam)
            - (self.loc / (2 * self.lam))
            * jnp.sqrt(4 * self.loc * self.lam * y + self.loc * self.loc * y * y)
        )
        key, subkey = jax.random.split(key)
        z = jax.random.uniform(subkey, shape=out_shape)
        return jnp.where(z <= (self.loc / (self.loc + x)), x, (self.loc**2) / x)

    @validate_sample
    def log_prob(self, value):
        logp = jnp.where(
            value <= 0,
            -jnp.inf,
            (
                logpow(self.lam / (2.0 * jnp.pi), 0.5)
                - logpow(value, 1.5)
                - (0.5 * self.lam / value * ((value - self.loc) / self.loc) ** 2)
            ),
        )
        return logp
