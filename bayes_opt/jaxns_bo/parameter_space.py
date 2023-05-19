from typing import Annotated, Union, Literal, List

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import vmap
from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.types import float_type
from jaxns.nested_sampling.prior import Prior, AbstractPrior, PriorModelGen, PriorModelType
from jaxns.nested_sampling.special_priors import Categorical
from pydantic import BaseModel, Field, validator
from scipy.optimize import least_squares

tfpd = tfp.distributions


class AffineBetaPrior(BaseModel):
    type: Literal['affine_beta_prior'] = 'affine_beta_prior'
    lower: float = Field(
        description="The greatest lower bound of interval. Inclusive.",
        example=0.1
    )
    upper: float = Field(
        description="The least upper bound of interval. Inclusive.",
        example=5.5
    )
    mode: float = Field(
        description="The mode of the prior.",
        example=2.5
    )
    uncert: float = Field(
        description="The uncertainty of the prior. Set to np.inf for the uniform prior over (lower, upper).",
        example=2.
    )


class DiscreteAffineBetaPrior(BaseModel):
    type: Literal['discrete_affine_beta_prior'] = 'discrete_affine_beta_prior'

    lower: int = Field(
        description="The greatest lower bound of interval. Inclusive.",
        example=0
    )
    upper: int = Field(
        description="The least upper bound of interval. Inclusive.",
        example=5
    )
    mode: float = Field(
        description="The mode of the prior. Can be a float.",
        example=2.5
    )
    uncert: float = Field(
        description="The uncertainty of the prior. Set to np.inf for the uniform prior over (lower, upper). Can be a float.",
        example=2.
    )


class CategoricalPrior(BaseModel):
    type: Literal['categorical_prior'] = 'categorical_prior'
    categories: List[str] = Field(
        description="The categories of the parameter.",
        examples=['a', 'b', 'c']
    )
    probs: List[float] = Field(
        description="The probabilities of categories. Need not be normalised.",
        example=[0.1, 0.3, 0.6]
    )


ParamPrior = Annotated[
    Union[AffineBetaPrior, DiscreteAffineBetaPrior, CategoricalPrior],
    Field(
        description='The parameter prior, which defines the domain.',
        discriminator='type'
    )
]


class Parameter(BaseModel):
    name: str = Field(
        description="The name of the parameter",
        example='price'
    )
    prior: ParamPrior


class ParameterSpace(BaseModel):
    parameters: List[Parameter] = Field(
        description='The parameters of the problem.',
        example=[
            Parameter(
                name='continuous',
                prior=AffineBetaPrior(
                    lower=0,
                    upper=5,
                    mode=4,
                    uncert=1
                )
            ),
            Parameter(
                name='integers',
                prior=DiscreteAffineBetaPrior(
                    lower=0,
                    upper=5,
                    mode=4,
                    uncert=jnp.inf
                )
            ),
            Parameter(
                name='categorical',
                prior=CategoricalPrior(
                    categories=['a', 'b', 'c'],
                    probs=[1., 1., 1.]
                )
            )
        ]
    )

    @validator('parameters', always=True)
    def unique_parameters(cls, value):
        names = list(map(lambda param: param.name, value))
        if len(names) != len(set(names)):
            raise ValueError(f"parameter names must be unique. Got {names}.")
        return value


def infer_beta_parameters(mode: float, variance: float):
    if jnp.isinf(variance):
        return 1, 1

    def equations(params):
        a, b = params
        eq1 = (a - 1) / (a + b - 2) - mode
        eq2 = (a * b) / ((a + b) ** 2 * (a + b + 1)) - variance
        return (eq1, eq2)

    # Initial guess for a and b
    initial_guess = np.array([2.0, 2.0])

    # Define bounds for a and b
    lower_bounds = np.array([1.0, 1.0])
    upper_bounds = np.array([np.inf, np.inf])
    bounds = (lower_bounds, upper_bounds)

    # Solve the system of equations with bounds
    result = least_squares(equations, initial_guess, bounds=bounds)

    a, b = result.x
    return a, b


def translate_parameter(param: Parameter) -> AbstractPrior:
    prior = param.prior
    if isinstance(prior, AffineBetaPrior):
        scale = prior.upper - prior.lower
        shift = prior.lower
        mode = (prior.mode - shift) / scale
        uncert = prior.uncert / scale
        alpha, beta = infer_beta_parameters(mode, uncert ** 2)
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        scale_bij = tfp.bijectors.Scale(scale=jnp.asarray(scale))
        shift_bij = tfp.bijectors.Shift(shift=jnp.asarray(shift))
        bij = shift_bij(scale_bij)
        underlying_dist = tfpd.Beta(concentration0=alpha, concentration1=beta)
        return Prior(dist_or_value=bij(underlying_dist), name=param.name)
    elif isinstance(prior, DiscreteAffineBetaPrior):
        scale = prior.upper - prior.lower + 1
        shift = prior.lower
        mode = (prior.mode - shift) / scale
        uncert = prior.uncert / scale
        alpha, beta = infer_beta_parameters(mode, uncert ** 2)
        alpha = jnp.asarray(alpha, float_type)
        beta = jnp.asarray(beta, float_type)
        scale_bij = tfp.bijectors.Scale(scale=jnp.asarray(scale, float_type))
        shift_bij = tfp.bijectors.Shift(shift=jnp.asarray(shift, float_type))
        bij = shift_bij(scale_bij)
        underlying_dist = tfpd.Beta(concentration0=alpha, concentration1=beta)
        transformed_dist: tfpd.TransformedDistribution = bij(underlying_dist)
        # P(0) = (P(0.) + P(1.))/2
        # P(1) = (P(1.) + P(2.))/2
        integer_categories = jnp.arange(jnp.asarray(prior.lower, float_type), jnp.asarray(prior.upper + 2, float_type),
                                        dtype=float_type)
        #
        log_probs_edges = vmap(lambda v: transformed_dist.log_prob(value=v))(integer_categories)
        log_probs = ((LogSpace(log_probs_edges[1:]) + LogSpace(log_probs_edges[:-1])) * LogSpace(
            jnp.log(0.5))).log_abs_val

        return Categorical(parametrisation='gumbel_max', logits=log_probs,
                           name=param.name)  # TODO: use cdf after jaxns release
    elif isinstance(prior, CategoricalPrior):
        return Categorical(parametrisation='gumbel_max', probs=prior.probs, name=param.name)
    else:
        raise ValueError(f"Invalid prior {prior}")


def build_prior_model(parameter_space: ParameterSpace) -> PriorModelType:
    """
    Constructs a prior model given the parameter space.

    Args:
        parameter_space:

    Returns:

    """

    def prior_model() -> PriorModelGen:
        param_values = []
        for parameter in sorted(parameter_space.parameters, key=lambda param: param.name):
            x = yield translate_parameter(param=parameter)
            param_values.append(x)
        return tuple(param_values)

    return prior_model
