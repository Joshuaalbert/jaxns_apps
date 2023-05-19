from jax import vmap, random, numpy as jnp

from jaxns_bo.parameter_space import Parameter, AffineBetaPrior, translate_parameter, DiscreteAffineBetaPrior, \
    CategoricalPrior, ParameterSpace


def test_build_prior():
    param = Parameter(
        name='continuous',
        prior=AffineBetaPrior(
            lower=0,
            upper=5,
            mode=4,
            uncert=1
        )
    )
    prior = translate_parameter(param=param)
    print(prior)
    x = vmap(lambda key: prior.forward(random.uniform(key=key, shape=prior.base_shape)))(
        random.split(random.PRNGKey(42), 1000))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 5.)
    assert jnp.any(x < 1.)
    assert jnp.any(x > 4.)

    param = Parameter(
        name='integers',
        prior=DiscreteAffineBetaPrior(
            lower=0,
            upper=5,
            mode=4,
            uncert=jnp.inf
        )
    )
    prior = translate_parameter(param=param)
    print(prior)
    x = vmap(lambda key: prior.forward(random.uniform(key=key, shape=prior.base_shape)))(
        random.split(random.PRNGKey(42), 1000))
    # print(x)
    assert jnp.all(x >= 0)
    assert jnp.all(x <= 5)
    assert jnp.any(x == 0)
    assert jnp.any(x == 5)

    param = Parameter(
        name='categorical',
        prior=CategoricalPrior(
            categories=['a', 'b', 'c'],
            probs=[1., 1., 1.]
        )
    )
    prior = translate_parameter(param=param)
    print(prior)
    x = vmap(lambda key: prior.forward(random.uniform(key=key, shape=prior.base_shape)))(
        random.split(random.PRNGKey(42), 1000))
    # print(x)
    assert jnp.all(x >= 0)
    assert jnp.all(x <= 2)
    assert jnp.any(x == 0)
    assert jnp.any(x == 2)


def test_parameter_space():
    param1 = Parameter(
        name='continuous',
        prior=AffineBetaPrior(
            lower=0,
            upper=5,
            mode=4,
            uncert=1
        )
    )

    param2 = Parameter(
        name='integers',
        prior=DiscreteAffineBetaPrior(
            lower=0,
            upper=5,
            mode=4,
            uncert=jnp.inf
        )
    )

    param3 = Parameter(
        name='categorical',
        prior=CategoricalPrior(
            categories=['a', 'b', 'c'],
            probs=[1., 1., 1.]
        )
    )

    _ = ParameterSpace(parameters=[param1, param2, param3])
    try:
        _ = ParameterSpace(parameters=[param1, param1])
        assert False
    except ValueError as e:
        assert 'parameter names must be unique' in str(e)
