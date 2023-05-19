import numpy as np
from jax import random, numpy as jnp

from jaxns_bo.utils import sprint_hyper_cube, latin_hypercube_corr


def test_latin_hypercube():
    points = sprint_hyper_cube(random.PRNGKey(42), num_samples=10,
                               num_dim=2,
                               cube_scale=0.1)
    print(points)
    import pylab as plt
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


def test_latin_hypercube_corr():
    samples = latin_hypercube_corr(key=random.PRNGKey(42), num_samples=4, num_dim=2, corr_matrix=jnp.eye(2))
    import pylab as plt
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()

    print(np.cov(samples, rowvar=False))
