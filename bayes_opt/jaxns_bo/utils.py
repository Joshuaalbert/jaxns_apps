import inspect
from datetime import datetime, tzinfo
from typing import TypeVar, Type, Dict, Any

import numpy as np
from jax import random, numpy as jnp, vmap
from jax._src.lax.control_flow import fori_loop
from jax._src.lax_reference import erf
from pydantic import BaseModel


def latin_hypercube(key, num_samples, num_dim, cube_scale):
    """
    Sample from the latin-hypercube defined as the continuous analog of the discrete latin-hypercube.
    That is, if you partition each dimension into `num_samples` equal volume intervals then there is (conditionally)
    exactly one point in each interval. We guarantee that uniformity by randomly assigning the permutation of each dimension.
    The degree of randomness is controlled by `cube_scale`. A value of 0 places the sample at the center of the grid point,
    and a value of 1 places the value randomly inside the grid-cell.

    Args:
        key: PRNG key
        num_samples: number of samples in total to draw
        num_dim: number of dimensions in each sample
        cube_scale: The scale of randomness, in (0,1).

    Returns:
        latin-hypercube samples of shape [num_samples, num_dim]
    """
    key1, key2 = random.split(key, 2)
    cube_scale = jnp.clip(cube_scale, 0., 1.)
    samples = vmap(lambda key: random.permutation(key, num_samples))(random.split(key2, num_dim)).T
    samples += random.uniform(key1, shape=samples.shape, minval=0.5 - cube_scale / 2., maxval=0.5 + cube_scale / 2.)
    samples /= num_samples
    return samples


def sprint_hyper_cube(key, num_samples, num_dim, cube_scale):
    X = random.uniform(key, shape=(num_samples, num_dim))
    V = cube_scale * random.normal(key, shape=(num_samples, num_dim))

    def spring_force(x, y):
        dx = x - y

        return dx

    def update(i, state):
        (X, V) = state
        total_force = vmap(lambda x: jnp.sum(vmap(lambda y: spring_force(x, y))(X), axis=0))(X)
        V = V + 0.01 * total_force
        X = X + 0.01 * V
        X = jnp.clip(X, 0., 1.)
        return (X, V)

    (X, V) = fori_loop(lower=0, upper=60, body_fun=update, init_val=(X, V))
    return X


def latin_hypercube_corr(key, num_samples: int, num_dim: int, corr_matrix: jnp.ndarray):
    # Check if the correlation matrix is 2D and square
    assert corr_matrix.ndim == 2
    assert corr_matrix.shape[0] == corr_matrix.shape[1]
    assert corr_matrix.shape[0] == num_dim

    # Initialize the array that will hold the sample points
    result = np.zeros((num_samples, num_dim))

    # Generate an uncorrelated LHS
    for dim in range(num_dim):
        sample = np.random.uniform(low=0, high=1, size=num_samples)
        seq = np.random.permutation(num_samples)
        result[:, dim] = (seq + sample) / num_samples

    # Rank the values
    rank = result.argsort(axis=0)

    # Create standard normal variables
    def single_dim(key):
        perm_key, sample_key = random.split(key, 2)
        normal_variables = random.normal(key=sample_key,
                                         shape=(num_samples,))
        normal_variables = random.permutation(key=perm_key, x=jnp.sort(normal_variables))
        return normal_variables

    normal_variables = vmap(single_dim)(random.split(key, num_dim)).T
    # Use the Cholesky decomposition to introduce the correlation
    chol = jnp.linalg.cholesky(corr_matrix)

    # Multiply the normal variables by the Cholesky decomposition
    correlated_variables = jnp.dot(chol, normal_variables.T).T

    def transform_back(v):
        return (erf(v / np.sqrt(2)) + 1) / 2

    return transform_back(correlated_variables)


def example_from_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate example from schema and return as dict.

    Args:
        model: BaseModel

    Returns: dict of example
    """
    example = dict()
    properties = model.schema().get('properties', dict())
    for field in model.__fields__:
        # print(model, model.__fields__[field])
        if inspect.isclass(model.__fields__[field]):
            if issubclass(model.__fields__[field], BaseModel):
                example[field] = example_from_schema(model.__fields__[field])
                continue
            example[field] = None
        example[field] = properties[field].get('example', None)
        # print(field, example[field])
    return example


_T = TypeVar('_T')


def build_example(model: Type[_T]) -> _T:
    return model(**example_from_schema(model))


def set_datetime_timezone(dt: datetime, offset: str | tzinfo) -> datetime:
    """
    Replaces the datetime object's timezone with one from an offset.

    Args:
        dt: datetime, with out without a timezone set. If set, will be replaced.
        offset: tzinfo, or str offset like '-04:00' (which means EST)

    Returns:
        datetime with timezone set
    """
    if isinstance(offset, str):
        dt = dt.replace(tzinfo=None)
        return datetime.fromisoformat(f"{dt.isoformat()}{offset}")
    if isinstance(offset, tzinfo):
        return dt.replace(tzinfo=offset)
    raise ValueError(f"offset {offset} not understood.")


def set_utc_timezone(dt: datetime) -> datetime:
    return set_datetime_timezone(dt, '+00:00')


def current_utc() -> datetime:
    return set_utc_timezone(datetime.utcnow())
