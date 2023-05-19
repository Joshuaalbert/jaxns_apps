from jax import numpy as jnp

from jaxns_bo.experiment import Trial, ParamValue, FloatValue, IntValue, StrValue, OptimisationExperiment, \
    ObjectiveMeasurement
from jaxns_bo.parameter_space import Parameter, AffineBetaPrior, DiscreteAffineBetaPrior, CategoricalPrior, \
    ParameterSpace
from jaxns_bo.utils import current_utc


def test_optimisation_experiment():
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

    parameter_space = ParameterSpace(parameters=[param1, param2, param3])

    trials = [
        Trial(param_values=[
            ParamValue(name='continuous', value=FloatValue(value=1.)),
            ParamValue(name='integers', value=IntValue(value=1)),
            ParamValue(name='categorical', value=StrValue(value='a'))
        ])
    ]
    s = OptimisationExperiment(parameter_space=parameter_space, trials=trials)
    assert s == OptimisationExperiment.parse_raw(s.json())

    s.trials[0].objective_measurement = ObjectiveMeasurement(value=FloatValue(value=1.))
    s.trials[0].measurement_dt = current_utc()

    assert s == OptimisationExperiment.parse_raw(s.json())

    # Validation errors

    try:
        trials = [
            Trial(param_values=[
                ParamValue(name='continuous', value=FloatValue(value=1.)),
                ParamValue(name='integers', value=IntValue(value=1))
            ])
        ]
        _ = OptimisationExperiment(parameter_space=parameter_space, trials=trials)
        assert False
    except ValueError as e:
        assert "don't match param space" in str(e)
