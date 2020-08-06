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

"""Eve: a stencil toolchain in pure Python."""

from .version import __version__, __versioninfo__  # noqa isort:skip

from . import codegen, exceptions, traits, utils  # noqa: F401
from .concepts import (  # noqa: F401
    FieldKind,
    FrozenModel,
    FrozenNode,
    Model,
    Node,
    SourceLocation,
    UIDGenerator,
    VType,
    field,
    in_field,
    out_field,
    symbol_field,
    validator,
)
from .tree_utils import FindNodes  # noqa: F401
from .types import (  # noqa: F401
    Bool,
    Bytes,
    Enum,
    Float,
    Int,
    IntEnum,
    NegativeFloat,
    NegativeInt,
    PositiveFloat,
    PositiveInt,
    Str,
    StrEnum,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    classproperty,
)
from .utils import NOTHING  # noqa: F401
from .visitors import NodeModifier, NodeTranslator, NodeVisitor, PathNodeVisitor  # noqa: F401
