from typing import (
  ClassVar as _ClassVar,
)
from typing import (
  Iterable as _Iterable,
)
from typing import (
  Mapping as _Mapping,
)
from typing import (
  Optional as _Optional,
)
from typing import (
  Union as _Union,
)

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class DistanceMeasure(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
  __slots__ = ()
  UNKNOWN: _ClassVar[DistanceMeasure]
  COSINE: _ClassVar[DistanceMeasure]
  DOT_PRODUCT: _ClassVar[DistanceMeasure]
  EUCLIDEAN: _ClassVar[DistanceMeasure]

UNKNOWN: DistanceMeasure
COSINE: DistanceMeasure
DOT_PRODUCT: DistanceMeasure
EUCLIDEAN: DistanceMeasure

class Vector(_message.Message):
  __slots__ = ("uri", "vector")
  URI_FIELD_NUMBER: _ClassVar[int]
  VECTOR_FIELD_NUMBER: _ClassVar[int]
  uri: str
  vector: _containers.RepeatedScalarFieldContainer[float]
  def __init__(
    self, uri: _Optional[str] = ..., vector: _Optional[_Iterable[float]] = ...
  ) -> None: ...

class GetVectorsRequest(_message.Message):
  __slots__ = (
    "uris",
    "exclude_estimated_vectors",
    "model",
    "batch",
    "user_vector_type",
  )
  URIS_FIELD_NUMBER: _ClassVar[int]
  EXCLUDE_ESTIMATED_VECTORS_FIELD_NUMBER: _ClassVar[int]
  MODEL_FIELD_NUMBER: _ClassVar[int]
  BATCH_FIELD_NUMBER: _ClassVar[int]
  USER_VECTOR_TYPE_FIELD_NUMBER: _ClassVar[int]
  uris: _containers.RepeatedScalarFieldContainer[str]
  exclude_estimated_vectors: bool
  model: str
  batch: str
  user_vector_type: str
  def __init__(
    self,
    uris: _Optional[_Iterable[str]] = ...,
    exclude_estimated_vectors: bool = ...,
    model: _Optional[str] = ...,
    batch: _Optional[str] = ...,
    user_vector_type: _Optional[str] = ...,
  ) -> None: ...

class GetVectorsResponse(_message.Message):
  __slots__ = ("model", "batch", "distance_measure", "vectors")
  MODEL_FIELD_NUMBER: _ClassVar[int]
  BATCH_FIELD_NUMBER: _ClassVar[int]
  DISTANCE_MEASURE_FIELD_NUMBER: _ClassVar[int]
  VECTORS_FIELD_NUMBER: _ClassVar[int]
  model: str
  batch: str
  distance_measure: DistanceMeasure
  vectors: _containers.RepeatedCompositeFieldContainer[Vector]
  def __init__(
    self,
    model: _Optional[str] = ...,
    batch: _Optional[str] = ...,
    distance_measure: _Optional[_Union[DistanceMeasure, str]] = ...,
    vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ...,
  ) -> None: ...
