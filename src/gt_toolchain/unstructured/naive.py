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
from typing import Optional, List, Tuple, Union

from pydantic import Field, validator, root_validator  # noqa: F401

from eve.core import Node, SourceLocation, StrEnum, NodeVisitor

from . import common


@enum.unique
class LocationType(enum.IntEnum):
    Node = 0
    Edge = 1
    Face = 2


class Expr(Node):
    location_type: LocationType
    pass


class Stmt(Node):
    location_type: LocationType
    pass


class FieldAccessExpr(Expr):
    name: str
    offset: Tuple[bool, int]
    # TODO need to lookup the symbol table for the field's location type

    # @root_validator
    # def check_location_type(cls, values):
    #     left, right, loctype = values.get('left'), values.get('right'), values.get('location_type')
    #     if left.location_type is not loctype or right.location_type is not loctype:
    #         raise ValueError('Location type mismatch')
    #     return values


class LiteralExpr(Expr):
    value: str
    data_type: common.DataType


class AssignmentExpr(Expr):
    left: Expr
    right: Expr

    # @root_validator # TODO pre!
    # def check_location_type(cls, values):
    #     if not(values['left'] == values['right']):
    #         raise ValueError('Location type mismatch')

    #     if "location_type" not in values:
    #         values['location_type'] = values['left']
    #     else:
    #         if not(values['left'] == values['location_type']):
    #             raise ValueError('Location type mismatch')


class ReduceOverNeighbourExpr(Expr):
    operation: common.BinaryOperator
    right: Expr
    init: Expr
    right_location_type: LocationType

    # @root_validator
    # def check_location_type(cls, values):
    #     right, init, right_loctype = values.get('right'), values.get('init'), values.get('right_location_type')
    #     if right.location_type is not right_loctype and init.location_type is not right_loctype:
    #         raise ValueError('Location type mismatch')
    #     return values


class BlockStmt(Stmt):
    statements: List[Stmt]

    # @root_validator
    # def check_location_type(cls, values):
    #     statements, loctype = values.get('statements'), values.get('location_type')
    #     for stmt in statements:
    #         if(stmt.location_type is not loctype):
    #             raise ValueError('Location type mismatch')
    #     return values


class ExprStmt(Stmt):
    expr: AssignmentExpr

    # @root_validator
    # def check_location_type(cls, values):
    #     expr, loctype = values.get('expr'), values.get('location_type')
    #     if expr.location_type is not loctype:
    #         raise ValueError('Location type mismatch')
    #     return values


class UnstructuredField(Node):
    name: str
    location_type: LocationType
    data_type: common.DataType


class HorizontalLoop(Node):
    location_type: LocationType
    ast: BlockStmt

    # @root_validator
    # def check_location_type(cls, values):
    #     ast, loctype = values.get('ast'), values.get('location_type')
    #     if ast.location_type is not loctype:
    #         raise ValueError('Location type mismatch')
    #     return values


class ForK(Node):
    horizontal_loops: List[HorizontalLoop]
    loop_order: common.LoopOrder


class Stencil(Node):
    name: str
    k_loops: List[ForK]


class Computation(Node):
    params: List[UnstructuredField]
    stencils: List[Stencil]
