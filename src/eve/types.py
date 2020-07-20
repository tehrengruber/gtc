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


import enum
import sys
from typing import Any, Callable, Generator

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


if sys.version_info < (3, 8):
    from typing_extensions import Literal, TypedDict  # noqa: F401
else:
    from typing import Literal, TypedDict  # noqa: F401

AnyCallable = Callable[..., Any]
NoArgAnyCallable = Callable[[], Any]


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
