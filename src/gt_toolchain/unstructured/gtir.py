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

import enum
from typing import List, Optional

from devtools import debug  # noqa: F401
from pydantic import root_validator, validator

from eve import IntEnum, Node, Str, StrEnum
from gt_toolchain import common


@enum.unique
class LocationType(IntEnum):
    Vertex = 0
    Edge = 1
    Cell = 2
    NoLocation = 3


class Expr(Node):
    location_type: LocationType
    pass


class Stmt(Node):
    location_type: LocationType
    pass


class Literal(Expr):
    value: Str
    vtype: common.DataType


class NeighborChain(Node):
    elements: List[LocationType]

    @validator("elements")
    def not_empty(cls, elements):
        if len(elements) <= 1:
            raise ValueError("NeighborChain must contain at least two locations")
        return elements


@enum.unique
class ReduceOperator(StrEnum):
    """Reduction operator identifier."""

    ADD = "ADD"
    MUL = "MUL"
    MAX = "MAX"
    MIN = "MIN"


class NeighborReduce(Expr):
    operand: Expr
    op: ReduceOperator
    neighbors: NeighborChain

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["neighbors"].elements[-1] != values["operand"].location_type:
            raise ValueError("Location type mismatch")
        return values


class FieldAccess(Expr):
    name: Str  # via symbol table


class AssignStmt(Stmt):
    left: FieldAccess  # there are no local variables in gtir, only fields
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["left"].location_type != values["right"].location_type:
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        elif values["left"].location_type != values["location_type"]:
            raise ValueError("Location type mismatch")

        return values


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["left"].location_type != values["right"].location_type:
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        elif values["left"] != values["location_type"]:
            raise ValueError("Location type mismatch")

        return values


class VerticalDimension(Node):
    pass


class HorizontalDimension(Node):
    primary: LocationType
    secondary: Optional[NeighborChain]


class Dimensions(Node):
    horizontal: Optional[HorizontalDimension]
    vertical: Optional[VerticalDimension]
    # other: TODO


class UField(Node):
    name: Str
    vtype: common.DataType
    dimensions: Dimensions


class TemporaryField(UField):
    pass


class HorizontalLoop(Node):
    stmt: Stmt
    location_type: LocationType

    @root_validator(pre=True)
    def check_location_type(cls, values):
        # Don't infer here! The location type of the loop should always come from the frontend!
        if values["stmt"].location_type != values["location_type"]:
            raise ValueError("Location type mismatch")
        return values


class VerticalLoop(Node):
    # each statement inside a `with location_type` is interpreted as a full horizontal loop (see parallel model of SIR)
    # TODO maybe still wrap it in a HorizontalLoop and force providing the location_type there
    #      (because for Stmts we allow to deduce it)
    horizontal_loops: List[HorizontalLoop]
    loop_order: common.LoopOrder


class Stencil(Node):
    vertical_loops: List[VerticalLoop]
    declarations: Optional[List[TemporaryField]]


class Computation(Node):
    name: Str
    params: List[UField]
    stencils: List[Stencil]
