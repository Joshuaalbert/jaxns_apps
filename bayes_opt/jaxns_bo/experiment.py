from datetime import datetime
from typing import Literal, List
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from jaxns_bo.parameter_space import ParameterSpace
from jaxns_bo.utils import build_example, current_utc


class FloatValue(BaseModel):
    type: Literal['float_value'] = 'float_value'
    value: float


class IntValue(BaseModel):
    type: Literal['int_value'] = 'int_value'
    value: int


class StrValue(BaseModel):
    type: Literal['str_value'] = 'str_value'
    value: str


class ParamValue(BaseModel):
    name: str = Field(
        description='Name of param value, matches the parameter name.',
        example='price'
    )
    value: FloatValue | IntValue | StrValue = Field(
        description='Value of parameters.',
        discriminator='type',
        example=FloatValue(value=1.)
    )


class ObjectiveMeasurement(BaseModel):
    value: FloatValue = Field(
        description='Value of objective function.',
        example=FloatValue(value=1.)
    )


class Trial(BaseModel):
    trial_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description='UUID for this trial.',
        example=str(uuid4())
    )
    create_dt: datetime = Field(
        default_factory=current_utc,
        description='The datetime the param_value was determined.',
        example=current_utc()
    )
    measurement_dt: datetime | None = Field(
        default=None,
        description='The datetime the objective_measurement was determined. Should be None if no measurement.',
        example=None
    )
    param_values: List[ParamValue] = Field(
        description="The parameter value for a single trial.",
        example=[build_example(ParamValue)]
    )
    objective_measurement: ObjectiveMeasurement | None = Field(
        default=None,
        description="The measurement of trial objective function. May be None, until measurement acquired.",
        example=None
    )


class OptimisationExperiment(BaseModel):
    experiment_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description='UUID for this experiment.',
        example=str(uuid4())
    )
    parameter_space: ParameterSpace = Field(
        description='The parameter space that defines this experiment.',
        example=build_example(ParameterSpace)
    )
    trials: List[Trial] = Field(
        description="The list of trials that define the sequence of this experiment.",
        example=[build_example(Trial)]
    )

    @validator('trials', always=True)
    def ensure_parameters_match_space(cls, value, values):
        parameter_space: ParameterSpace = values['parameter_space']
        names = list(map(lambda param: param.name, parameter_space.parameters))
        for trial in value:
            _names = list(map(lambda param_value: param_value.name, trial.param_values))
            if set(_names) != set(names):
                raise ValueError(f"trial param_values {_names} don't match param space {names}.")
        return value
