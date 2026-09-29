# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pydantic model to dpsynth domain adapter.

Provides three core functions (matching `dpsynth.adapters.protobuf`) to bridge
flat `pydantic.BaseModel` schemas with `dpsynth` mechanisms:

- `infer_domain`: Infers a `dict[str, domain.AttributeType]` from a `BaseModel`
  class using field type annotations (`int`, `float`, `bool`, `Enum`, `Literal`)
  and `pydantic.Field(ge=..., le=..., gt=..., lt=...)` bounds metadata. Optional
  fields (`T | None`) set `clip_to_range=False` for numeric attributes and
  prepend `"None"` at index 0 for categorical attributes.
- `to_tuple`: Converts a `BaseModel` instance to a tuple of field values.
- `from_tuple`: Reconstructs a `BaseModel` instance from a sequence of values.
"""

from collections.abc import Mapping, Sequence
import enum
import inspect
import math
import types
import typing  # for typing.get_origin and typing.get_args
from typing import Any, Literal, TypeVar

from dpsynth import domain
import pydantic
from pydantic.fields import annotated_types
from pydantic.fields import FieldInfo  # pylint: disable=g-importing-member


def _get_base_type(annotation: type[Any]) -> tuple[bool, type[Any]]:
  """Determines the base type of a type annotation and if it is optional."""

  if annotation is types.NoneType:
    raise ValueError("Unexpected None type annotation.")

  is_optional = annotation == (annotation | None)

  if is_optional:
    # It's a Union like `int | None`. We need the non-None part.
    args = typing.get_args(annotation)
    non_none_args = [arg for arg in args if arg is not types.NoneType]
    assert len(non_none_args) == 1, "Union must have exactly one non-None type."
    annotation = non_none_args[0]

  if not (
      annotation in (int, float, str, bool)
      or (inspect.isclass(annotation) and issubclass(annotation, enum.Enum))
      or typing.get_origin(annotation) is Literal
  ):
    raise ValueError(f"Unexpected type annotation: {annotation}.")

  return is_optional, annotation


def _numerical_attribute_from_field_info(
    field_info: FieldInfo,
) -> domain.NumericalAttribute:
  """Infers a NumericalAttribute from a pydantic FieldInfo."""
  # NumericalAttribute uses a convention where both the min_value and max_value
  # are assumed to be inclusive.  More general bounds are supported by the
  # pydantic metadata, so we convert to the expected representation here.
  optional, base_type = _get_base_type(field_info.annotation)  # pyrefly: ignore[bad-argument-type]

  lower_bound = upper_bound = None
  for meta in field_info.metadata:
    match type(meta):
      case annotated_types.Ge:
        lower_bound = meta.ge
      case annotated_types.Le:
        upper_bound = meta.le
      case annotated_types.Gt:
        if base_type is int:
          lower_bound = meta.gt + 1
        else:
          lower_bound = math.nextafter(meta.gt, math.inf)
      case annotated_types.Lt:
        if base_type is int:
          upper_bound = meta.lt - 1
        else:
          upper_bound = math.nextafter(meta.lt, -math.inf)
      case _:
        continue
  if lower_bound is None or upper_bound is None:
    raise ValueError("Must specify lower and upper bounds for numeric fields.")

  return domain.NumericalAttribute(
      min_value=lower_bound,  # pyrefly: ignore[unexpected-keyword]
      max_value=upper_bound,  # pyrefly: ignore[unexpected-keyword]
      clip_to_range=not optional,  # pyrefly: ignore[unexpected-keyword]
      dtype=base_type.__name__,  # pyrefly: ignore[unexpected-keyword]
  )


def _categorical_attribute_from_field_info(
    field_info: FieldInfo,
) -> domain.CategoricalAttribute:
  """Infers a CategoricalAttribute from a pydantic FieldInfo."""
  optional, base_type = _get_base_type(field_info.annotation)  # pyrefly: ignore[bad-argument-type]
  if inspect.isclass(base_type) and issubclass(base_type, enum.Enum):
    possible_values = [str(e.value) for e in base_type]
  elif typing.get_origin(base_type) is Literal:
    possible_values = [str(v) for v in typing.get_args(base_type)]
  elif base_type is bool:
    possible_values = [str(False), str(True)]
  else:
    raise ValueError(f"Unexpected type annotation: {base_type}.")

  if optional:
    possible_values = [str(None)] + possible_values

  return domain.CategoricalAttribute(
      possible_values=possible_values, out_of_domain_index=0  # pyrefly: ignore[unexpected-keyword]
  )


def infer_domain(
    model_cls: type[pydantic.BaseModel],
) -> dict[str, domain.CategoricalAttribute | domain.NumericalAttribute]:
  """Infers the domain of a pydantic model."""

  attributes = {}
  for name, meta in model_cls.model_fields.items():
    _, base_type = _get_base_type(meta.annotation)  # pyrefly: ignore[bad-argument-type]
    if base_type in (int, float):
      attributes[name] = _numerical_attribute_from_field_info(meta)
    elif base_type is bool:
      attributes[name] = _categorical_attribute_from_field_info(meta)
    elif inspect.isclass(base_type) and issubclass(base_type, enum.Enum):
      attributes[name] = _categorical_attribute_from_field_info(meta)
    elif typing.get_origin(base_type) is Literal:
      attributes[name] = _categorical_attribute_from_field_info(meta)
    else:
      raise ValueError(f"Unexpected type annotation: {base_type}.")
  return attributes


RecordT = TypeVar("RecordT", bound=pydantic.BaseModel)


def to_tuple(
    record: RecordT,
    *,
    schema: Mapping[str, domain.AttributeType] | None = None,
) -> tuple[Any, ...]:
  """Converts a pydantic model instance to a tuple of field values."""
  schema = schema or infer_domain(type(record))
  row = record.model_dump(mode="json")
  result = []
  for col, attr in schema.items():
    val = row[col]
    if isinstance(attr, domain.CategoricalAttribute):
      val = str(val)
    elif isinstance(attr, domain.NumericalAttribute) and val is None:
      # the NumericalAttribute contract, but currently doesn't.
      val = attr.min_value
    result.append(val)
  return tuple(result)


def from_tuple(
    values: Sequence[Any],
    model_cls: type[RecordT],
    *,
    schema: Mapping[str, domain.AttributeType] | None = None,
) -> RecordT:
  """Converts a sequence of field values back to a pydantic model instance."""
  schema = schema or infer_domain(model_cls)

  def _coerce(attr, v):
    if v in (None, "None"):
      return None
    if isinstance(attr, domain.NumericalAttribute):
      if math.isnan(v):
        return None
      return round(float(v)) if attr.dtype == "int" else v
    return v

  kwargs = {k: _coerce(a, v) for (k, a), v in zip(schema.items(), values)}
  return model_cls(**kwargs)  # pyrefly: ignore[bad-unpacking]
