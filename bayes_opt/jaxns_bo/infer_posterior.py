# for Gaussian processes this is important
from jax.config import config
from jaxns.nested_sampling.model import Model
from jaxns.nested_sampling.nested_sampler import ExactNestedSampler
from jaxns.nested_sampling.prior import Prior
from jaxns.nested_sampling.types import TerminationCondition

config.update("jax_enable_x64", True)

import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions
tfpk = tfp.math.psd_kernels

from jaxns import marginalise_dynamic
from jax.scipy.linalg import solve_triangular
from jax import random
from jax import numpy as jnp
import pylab as plt
import numpy as np



from typing import Type


def run_for_kernel(kernel: Type[tfpk.PositiveSemidefiniteKernel]):
    print(("Working on Kernel: {}".format(kernel.__class__.__name__)))

    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        # U, S, Vh = jnp.linalg.svd(cov)
        log_det = jnp.sum(jnp.log(jnp.diag(L)))  # jnp.sum(jnp.log(S))#
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        # U S Vh V 1/S Uh
        # pinv = (Vh.T.conj() * jnp.where(S!=0., jnp.reciprocal(S), 0.)) @ U.T.conj()
        maha = dx @ dx  # dx @ pinv @ dx#solve_triangular(L, dx, lower=True)
        log_likelihood = -0.5 * x.size * jnp.log(2. * jnp.pi) - log_det - 0.5 * maha
        return log_likelihood

    def log_likelihood(uncert, l, sigma):
        """
        P(Y|sigma, half_width) = N[Y, f, K]
        Args:
            sigma:
            l:

        Returns:

        """
        K = kernel(amplitude=sigma, length_scale=l).matrix(X, X)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return log_normal(Y_obs, mu, K + data_cov)

    def predict_f(uncert, l, sigma):
        K = kernel(amplitude=sigma, length_scale=l).matrix(X, X)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return mu + K @ jnp.linalg.solve(K + data_cov, Y_obs)

    def predict_fvar(uncert, l, sigma):
        K = kernel(amplitude=sigma, length_scale=l).matrix(X, X)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return jnp.diag(K - K @ jnp.linalg.solve(K + data_cov, K))

    # Build the model

    def prior_model():
        l = yield Prior(tfpd.Uniform(0., 2.), name='l')
        uncert = yield Prior(tfpd.HalfNormal(1.), name='uncert')
        sigma = yield Prior(tfpd.Uniform(0., 2.), name='sigma')
        return uncert, l, sigma

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    model.sanity_check(random.PRNGKey(0), S=100)

    # Create the nested sampler class. In this case without any tuning.
    exact_ns = ExactNestedSampler(model=model, num_live_points=model.U_ndims * 100, max_samples=1e6,
                                  uncert_improvement_patience=2)

    termination_reason, state = exact_ns(random.PRNGKey(42), term_cond=TerminationCondition(live_evidence_frac=1e-4))
    results = exact_ns.to_results(state, termination_reason)

    exact_ns.summary(results)
    exact_ns.plot_diagnostics(results)
    exact_ns.plot_cornerplot(results)

    predict_f = marginalise_dynamic(random.PRNGKey(42), results.samples, results.log_dp_mean,
                                    results.ESS, predict_f)

    predict_fvar = marginalise_dynamic(random.PRNGKey(42), results.samples, results.log_dp_mean,
                                       results.ESS, predict_fvar)

    plt.scatter(X[:, 0], Y_obs, label='data')
    plt.plot(X[:, 0], Y, label='underlying')
    plt.plot(X[:, 0], predict_f, label='marginalised')
    plt.plot(X[:, 0], predict_f + jnp.sqrt(predict_fvar), ls='dotted',
             c='black')
    plt.plot(X[:, 0], predict_f - jnp.sqrt(predict_fvar), ls='dotted',
             c='black')
    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.legend()
    plt.show()

    return results.log_Z_mean, results.log_Z_uncert



# Let us compare these models.

logZ_rbf, logZerr_rbf = run_for_kernel(tfpk.ExponentiatedQuadratic)
logZ_m12, logZerr_m12 = run_for_kernel(tfpk.MaternOneHalf)
logZ_m32, logZerr_m32 = run_for_kernel(tfpk.MaternThreeHalves)

plt.errorbar(['rbf', 'm12', 'm32'], [logZ_rbf, logZ_m12, logZ_m32], [logZerr_rbf, logZerr_m12, logZerr_m32])
plt.ylabel("log Z")
plt.legend()
plt.show()