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
import ast
import inspect
from typing import Any, Dict, ForwardRef, Generic, List, Optional, TypeVar, Union

import typing_inspect

import eve.types
import gtc.common as common
from eve import Node, UIDGenerator

from . import ast_node_matcher as anm

# builtins
# todo: is this the right way?
from .built_in_types import Field, Local, Location, Mesh, TemporaryField


Vertex = common.LocationType.Vertex
Edge = common.LocationType.Edge
Cell = common.LocationType.Cell

# template is a 1-to-1 mapping from context and python ast node to gt4py ast node. context is encoded in the field types
# all understood sementic is encoded in the structure


class GTScriptAstNode(Node):
    pass


class Statement(GTScriptAstNode):
    pass


class Expr(GTScriptAstNode):
    pass


class Name(Expr):
    id: str


class IterationOrder(GTScriptAstNode):
    order: str


# todo: use type parameter see https://github.com/samuelcolvin/pydantic/pull/595
# T = TypeVar('T')
# class Constant(GT4PyAstNode, Generic[T]):
#    value: T


class Constant(Expr):
    # todo: due to automatic conversion in pydantic str must be at the end. evaluate usage of StrictStr etc.
    value: Union[int, float, type(None), str]


class Interval(GTScriptAstNode):
    start: Constant  # todo: use Constant[Union[int, str, type(None)]]
    stop: Constant


# todo: allow interval(...) by introducing Optional(captures={...}) placeholder
# Optional(captures={start=0, end=None})


class LocationSpecification(GTScriptAstNode):
    name: Name
    location_type: str


# todo: proper cannonicalization (CanBeCanonicalizedTo[Subscript] ?)
class SubscriptSingle(Expr):
    value: Name
    index: str


SubscriptMultiple = ForwardRef("SubscriptMultiple")


class SubscriptMultiple(Expr):
    value: Name
    indices: List[Union[Name, SubscriptSingle, SubscriptMultiple]]


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


class Call(Expr):
    args: List[Expr]
    func: str


class LocationComprehension(GTScriptAstNode):
    target: Name
    iter: Call


class Generator(Expr):
    generators: List[LocationComprehension]
    elt: Expr


class Assign(Statement):
    target: Union[Name, SubscriptSingle, SubscriptMultiple]  # todo: allow subscript
    value: Expr


Stencil = ForwardRef("Stencil")


class Stencil(GTScriptAstNode):
    iteration_spec: List[Union[IterationOrder, LocationSpecification, Interval]]
    body: List[Union[Statement, Stencil]]  # todo: stencil only allowed non-canonicalized


# class Attribute(GT4PyAstNode):
#    attr: str
#    value: Union[Attribute, Name]
#
#    @staticmethod
#    def template():
#        return ast.Attribute(attr=Capture("attr"), value=Capture("value"))


class Pass(Statement):
    pass


class Argument(GTScriptAstNode):
    name: str
    type: Union[Name, Union[SubscriptMultiple, SubscriptSingle]]
    # is_keyword: bool


# class Call(Generic[T]):
#    name: str
#    return_type: T
#    arg_types: Ts
#    args: List[Expr]


class Computation(GTScriptAstNode):
    name: str
    arguments: List[Argument]
    stencils: List[Stencil]
    # stencils: List[Union[Stencil[Stencil[Statement]], Stencil[Statement]]]
