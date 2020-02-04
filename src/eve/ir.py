# -*- coding: utf-8 -*-
#
# Eve Toolchain - GridTools Project
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import enum
from typing import Optional

from pydantic import Field, validator

from .core import Node, SourceLocation, StrEnum


class AssignmentKind(StrEnum):
    """Kind of assignment: plain or combined with operations."""

    PLAIN = "="
    ADD = "+="
    SUB = "-="
    MUL = "*="
    DIV = "/="


@enum.unique
class UnaryOperator(StrEnum):
    """Kind of assignment: plain or combined with operations."""

    POS = "+"
    NEG = "-"


@enum.unique
class BinaryOperator(StrEnum):
    """Binary operator identifier."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


@enum.unique
class DataType(enum.IntEnum):
    """Data type identifier."""

    INVALID = -1
    AUTO = 0
    BOOLEAN = 1
    INT32 = 101
    UINT32 = 102
    FLOAT32 = 201
    FLOAT64 = 202


class Expr(Node):
    pass


class LiteralExpr(Expr):
    value: str = Field(..., description="Value definition")
    data_type: DataType = Field(DataType.AUTO, description="Value data type")
    loc: Optional[SourceLocation]


class UnaryOpExpr(Expr):
    op: UnaryOperator = Field(..., description="Operator")
    operand: Expr = Field(..., description="Expression affected by the operator")
    loc: Optional[SourceLocation]


class BinaryOpExpr(Expr):
    op: BinaryOperator = Field(..., description="Operator")
    left: Expr = Field(..., description="Left-hand side of the expression")
    right: Expr = Field(..., description="Right-hand side of the expression")
    loc: Optional[SourceLocation]


class TernaryOpExpr(Expr):
    cond: Expr = Field(..., description="Condition")
    left: Expr = Field(..., description="Left-hand side of the expression")
    right: Expr = Field(..., description="Right-hand side of the expression")
    loc: Optional[SourceLocation]


class AssignmentExpr(Expr):
    kind: AssignmentKind
    left: Expr = Field(..., description="Left-hand side")
    right: Expr = Field(..., description="Right-hand side")
    loc: Optional[SourceLocation]
