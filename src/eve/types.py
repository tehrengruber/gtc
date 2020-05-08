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

"""Definitions of foundation classes."""


import enum
from typing import Any, Generator

from boltons.typeutils import classproperty
from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr
from pydantic.typing import AnyCallable
from typing_extensions import Literal  # noqa: F401


ClassProperty = classproperty  # noqa: F401
Bool = StrictBool  # noqa: F401
Bytes = bytes  # noqa: F401
Float = StrictFloat  # noqa: F401
Int = StrictInt  # noqa: F401
Str = StrictStr  # noqa: F401


#: Typing hint for `__get_validators__()` methods (defined but not exported in `pydantic.typing`)
CallableGenerator = Generator[AnyCallable, None, None]


class Enum(enum.Enum):
    """Basic :class:`enum.Enum` subclass with strict type validation."""

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls._strict_type_validator
        if hasattr(super(), "__get_validators__"):
            yield from super().__get_validators__()

    @classmethod
    def _strict_type_validator(cls, v: Any) -> enum.Enum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v


class IntEnum(enum.IntEnum):
    """Basic :class:`enum.IntEnum` subclass with strict type validation."""

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls._strict_type_validator
        if hasattr(super(), "__get_validators__"):
            yield from super().__get_validators__()

    @classmethod
    def _strict_type_validator(cls, v: Any) -> enum.IntEnum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v


class StrEnum(str, enum.Enum):
    """Basic :class:`enum.Enum` subclass with strict type validation and supporting string operations."""

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls._strict_type_validator
        if hasattr(super(), "__get_validators__"):
            yield from super().__get_validators__()

    @classmethod
    def _strict_type_validator(cls, v: Any) -> "StrEnum":
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v

    def __str__(self) -> str:
        return self.value
