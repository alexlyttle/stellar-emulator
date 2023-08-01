import jax.numpy as jnp
from jax import lax, random
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes


class Salpeter(Distribution):
    """Salpeter initial mass function (IMF).
    
    M ~ IMF(alpha)

    IMF(alpha) = norm * M**(- alpha)

    Normalised from 'low' to 'high'.

    Args:
        alpha (float): Exponent of the IMF.
        low (float): Lower bound of the IMF. Default is 0.1.
        high (float): Upper bound of the IMG. Default is infinity.
        validate_args (bool): Whether to validate arguments. Default is None.
    """
    arg_constraints = {
        "alpha": constraints.positive,
        "low": constraints.positive,
        "high": constraints.positive
    }
    reparametrized_params = ["alpha", "low", "high"]

    def __init__(self, alpha, *, low=0.1, high=float("inf"), validate_args=None):
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(low),
            jnp.shape(high),
        )
        self.alpha, self.low, self.high = promote_shapes(alpha, low, high)
        self.beta = 1 - self.alpha
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        """Disttribution support."""
        return constraints.interval(self.low, self.high)

    @property
    def _norm(self):
        return (self.high**self.beta - self.low**self.beta) / self.beta

    def log_prob(self, value):
        """Log probability density of value given distribution."""
        log_m = jnp.log(self._norm)
        log_p = - self.alpha * jnp.log(value)
        return jnp.where((value <= self.high) & (value > self.low), log_p - log_m, -jnp.inf)

    def cdf(self, value):
        """Cumulative probability density for value given distribution."""
        return (value**self.beta - self.low**self.beta) / self.beta / self._norm
    
    def icdf(self, u):
        """Inverse cumulative probability density function."""
        return (self.beta * self._norm * u + self.low**self.beta)**(1/self.beta)
    
    def sample(self, key, sample_shape=()):
        """Draw sample from distribution."""
        shape = sample_shape + self.batch_shape
        minval = jnp.finfo(jnp.result_type(float)).tiny
        u = random.uniform(key, shape, minval=minval)
        return self.icdf(u)
