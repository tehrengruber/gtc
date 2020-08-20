import ast

from typing import ForwardRef, Union, Dict, List, Optional, Any, Generic, TypeVar
import inspect
from eve import Node, UIDGenerator
import gtc.common as common
from . import ast_node_matcher as anm

import typing_inspect

# builtins
# todo: is this the right way?
from .built_in_types import Mesh, Field, TemporaryField, Location, Local
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
#T = TypeVar('T')
#class Constant(GT4PyAstNode, Generic[T]):
#    value: T

class Constant(Expr):
    value: Union[int, float, type(None), str] # note: due to automatic conversion in pydantic str must be at the end


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


SubscriptMultiple = ForwardRef('SubscriptMultiple')


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
    target: Name  # todo: allow subscript
    value: Expr


Stencil = ForwardRef('Stencil')

class Stencil(GTScriptAstNode):
    iteration_spec: List[Union[IterationOrder, LocationSpecification, Interval]]
    body: List[Union[Statement, Stencil]] # todo: stencil only allowed non-canonicalized

#class Attribute(GT4PyAstNode):
#    attr: str
#    value: Union[Attribute, Name]
#
#    @staticmethod
#    def template():
#        return ast.Attribute(attr=Capture("attr"), value=Capture("value"))


class Pass(Statement):
    @staticmethod
    def template():
        return ast.Pass()


class Argument(GTScriptAstNode):
    name: str
    type: Union[Name, Union[SubscriptMultiple, SubscriptSingle]]
    #is_keyword: bool

#class Call(Generic[T]):
#    name: str
#    return_type: T
#    arg_types: Ts
#    args: List[Expr]

class Computation(GTScriptAstNode):
    name: str
    arguments: List[Argument]
    stencils: List[Stencil]
    #stencils: List[Union[Stencil[Stencil[Statement]], Stencil[Statement]]]
