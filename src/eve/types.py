# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Definitions of general infrastructure."""


import collections.abc
import dataclasses
import enum
import json
import typing
from typing import Any, Callable, Dict, Generator, Mapping, Optional, Type, TypeVar, Union

import boltons.typeutils
import pydantic
from pydantic import (  # noqa: F401
    NegativeFloat,
    NegativeInt,
    PositiveFloat,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    validator,
)


if typing.TYPE_CHECKING:
    from pydantic.dataclasses import DataclassT, DataclassType

    # Typing proxies
    ModelclassT = TypeVar("ModelclassT", bound="ModelclassType")

    class ModelclassType(DataclassType):
        __dataclass_fields__: Dict[str, Any]

        def to_dict(self: "ModelclassT") -> Dict[str, Any]:
            ...

        def to_json(self: "ModelclassT") -> str:
            ...

        @classmethod
        def schema(cls: Type["ModelclassT"]) -> Dict[str, Any]:
            ...

        @classmethod
        def schema_json(cls: Type["ModelclassT"]) -> str:
            ...


def field(
    *,
    default: Union[Any, dataclasses._MISSING_TYPE] = dataclasses.MISSING,
    default_factory: Union[Callable, dataclasses._MISSING_TYPE] = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,  # noqa: A002
    hash: Optional[bool] = None,  # noqa: A002
    compare: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
    schema_info: Optional[Mapping[str, Any]] = None,
) -> dataclasses.Field:

    assert default is dataclasses.MISSING or default_factory is dataclasses.MISSING

    return dataclasses.field(  # type: ignore
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def modelclass(
    class_: Optional[Type] = None,
    *,
    # dataclass params
    init: bool = True,
    repr: bool = True,  # noqa: A002
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    # pydantic params
    config: Optional[Union[Type, Dict[str, Any]]] = None,
) -> Union[Callable[[Type], "ModelclassType"], "ModelclassType"]:

    if config is None:
        config_cls: Type = _DefaultPydanticConfig
    else:
        if isinstance(config, collections.abc.Mapping):
            config_cls = type("_PydanticConfig", (object,), config)
        elif not isinstance(config, type):
            raise TypeError(f"Invalid config class ({config})")

        for name, value in _DefaultPydanticConfig.__dict__.items():
            if not name.startswith("_") and not hasattr(config_cls, name):
                setattr(config_cls, name, value)

    assert isinstance(config_cls, type)

    def _wrapper(cls_: Type) -> "ModelclassType":
        model_cls = pydantic.dataclasses.dataclass(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            config=config_cls,
        )(cls_)

        def _typedclass_to_dict(self: "DataclassT") -> Dict[str, Any]:
            return typing.cast(Dict[str, Any], pydantic.json.pydantic_encoder(self))

        model_cls.to_dict = _typedclass_to_dict  # type: ignore

        def _typedclass_to_json(self: "DataclassT", *, indent: int = 4) -> str:
            return json.dumps(self, indent=indent, default=pydantic.json.pydantic_encoder)

        model_cls.to_json = _typedclass_to_json  # type: ignore

        model_cls.schema = model_cls.__pydantic_model__.schema  # type: ignore
        model_cls.schema_json = model_cls.__pydantic_model__.schema_json  # type: ignore

        return typing.cast("ModelclassType", model_cls)

    return _wrapper(class_) if class_ else _wrapper


class _DefaultPydanticConfig:
    extra = "forbid"
    arbitrary_types_allowed = True


classproperty = boltons.typeutils.classproperty


#: :class:`bool subclass for strict field definition
Bool = StrictBool  # noqa: F401
#: :class:`bytes subclass for strict field definition
Bytes = bytes  # noqa: F401
#: :class:`float` subclass for strict field definition
Float = StrictFloat  # noqa: F401
#: :class:`int` subclass for strict field definition
Int = StrictInt  # noqa: F401
#: :class:`str` subclass for strict field definition
Str = StrictStr  # noqa: F401


#: Typing hint for `__get_validators__()` methods (defined but not exported in `pydantic.typing`)
PydanticCallableGenerator = Generator[pydantic.typing.AnyCallable, None, None]


class Enum(enum.Enum):
    """Basic :class:`enum.Enum` subclass with strict type validation."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> enum.Enum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v


class IntEnum(enum.IntEnum):
    """Basic :class:`enum.IntEnum` subclass with strict type validation."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> enum.IntEnum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v


class StrEnum(str, enum.Enum):
    """Basic :class:`enum.Enum` subclass with strict type validation and supporting string operations."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> "StrEnum":
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v

    def __str__(self) -> str:
        assert isinstance(self.value, str)
        return self.value
