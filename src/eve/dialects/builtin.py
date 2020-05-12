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

"""Definitions of builtin ("") dialect nodes and vtypes."""

from .concepts import Dialect, VType


class BuiltinDialect(Dialect):
    name: ClassVar[str] = ""


@BuiltinDialect.register
class BuiltinVType(VType):
    pass


class NoneVType(BuiltinVType):
    name: ClassVar[str] = "none"


class BooleanVType(BuiltinVType):
    name: ClassVar[str] = "boolean"


class IndexVType(BuiltinVType):
    name: ClassVar[str] = "index"


class IntegerVType(BuiltinVType):
    name: ClassVar[str] = "name"


# complex, float, integer, boolean,
# index, memref
# tuple, struct, tensor
# singleton

# python
# complex > real > integral > bool


# dtypes

# b	boolean
# i	signed integer
# u	unsigned integer
# f	floating-point
# c	complex floating-point
# m	timedelta
# M	datetime
# O	object
# S	(byte-)string
# U	Unicode
# V	void

# standard-type ::=     complex-type
#                     | float-type
#                     | function-type
#                     | index-type
#                     | integer-type
#                     | memref-type
#                     | none-type
#                     | tensor-type
#                     | tuple-type
#                     | vector-type
