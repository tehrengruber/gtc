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
from typing import List, Optional, Union

from pydantic import root_validator

from eve import IntEnum, Node, Str
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


class FieldAccess(Expr):
    name: Str
    # TODO verify that the tag exists in the location_type composite (reachable via symbol table)


class VarDecl(Stmt):
    name: Str
    init: Expr
    # TODO type etc


class Literal(Expr):
    value: Str


class VarAccess(Expr):
    name: Str  # via symbol table


class AssignStmt(Stmt):
    left: Union[FieldAccess, VarAccess]
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        # if values["left"].location_type != values["right"].location_type:
        #     raise ValueError("Location type mismatch")

        # if "location_type" not in values:
        #     values["location_type"] = values["left"].location_type
        # elif values["left"] != values["location_type"]:
        #     raise ValueError("Location type mismatch")
        if values["left"].location_type == values["right"].location_type:
            values["location_type"] = values["left"].location_type

        return values


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        # if values["left"].location_type != values["right"].location_type:
        #     raise ValueError("Location type mismatch")

        # if "location_type" not in values:
        #     values["location_type"] = values["left"].location_type
        # elif values["left"] != values["location_type"]:
        #     raise ValueError("Location type mismatch")

        # TODO we need to protect (but above protection doesn't work in neighbor loop for accessing a variable from the parent location)
        if values["left"].location_type == values["right"].location_type:
            values["location_type"] = values["left"].location_type

        return values


class NeighborLoop(Stmt):
    body_location_type: LocationType
    body: List[Stmt]


class SidCompositeEntry(Node):
    name: Str  # ensure field exists via symbol table


class SidComposite(Node):
    location_type: LocationType
    entries: List[SidCompositeEntry]


class NeighborChain(Node):
    chain: List[LocationType]


class Kernel(Node):
    # location_type: LocationType
    name: Str  # symbol table
    primary_connectivity: LocationType
    other_connectivities: Optional[List[NeighborChain]]
    primary_sid_composite: SidComposite
    other_sid_composites: Optional[List[SidComposite]]
    ast: List[Stmt]


class SidTag(Node):
    name: Str


class VerticalDimension(Node):
    pass


class SecondaryLocation(Node):
    chain: List[LocationType]


class USid(Node):
    name: Str
    dimensions: List[Union[LocationType, SecondaryLocation, VerticalDimension]]  # Set?


class Temporary(USid):
    name: Str
    dimensions: List[Union[LocationType, SecondaryLocation, VerticalDimension]]  # Set?


class Computation(Node):
    name: Str
    parameters: List[USid]
    temporaries: List[Temporary]
    kernels: List[Kernel]  # probably replace by ctrlflow ast (where Kernel is one CtrlFlowStmt)
