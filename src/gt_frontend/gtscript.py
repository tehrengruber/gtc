import ast

from typing import Union, Dict, List, Optional, Any, Generic, TypeVar
import inspect
from eve import Node, UIDGenerator
import gtc.common as common
from . import ast_node_matcher as anm
from .ast_node_matcher import Capture
import typing_inspect

# builtins
# todo: is this the right way?
from .built_in_types import Mesh, Field, TemporaryField, Location
Vertex = common.LocationType.Vertex
Edge = common.LocationType.Edge
Cell = common.LocationType.Cell

# template is a 1-to-1 mapping from context and python ast node to gt4py ast node. context is encoded in the field types
# all understood sementic is encoded in the structure

class GT4PyAstNode(Node):
    pass

class Statement(GT4PyAstNode):
    pass

class Expr(GT4PyAstNode):
    pass

class Name(Expr):
    id: str

    @staticmethod
    def template():
        return ast.Name(id=Capture("id"))

class IterationOrder(GT4PyAstNode):
    order: str

    @staticmethod
    def template():
        return ast.withitem(context_expr=ast.Call(func=ast.Name(id="computation"), args=[ast.Name(id=Capture("order"))]))


# todo: use type parameter see https://github.com/samuelcolvin/pydantic/pull/595
#T = TypeVar('T')
#class Constant(GT4PyAstNode, Generic[T]):
#    value: T

class Constant(GT4PyAstNode):
    value: Union[str, int, type(None)]

    @staticmethod
    def template():
        return ast.Constant(value=Capture("value"))


class Interval(GT4PyAstNode):
    start: Constant  # todo: use Constant[Union[int, str, type(None)]]
    stop: Constant

    @staticmethod
    def template():
        return ast.withitem(context_expr=ast.Call(func=ast.Name(id="interval"),
                                                  args=[Capture("start"), Capture("stop")]))

# todo: allow interval(...) by introducing Optional(captures={...}) placeholder
# Optional(captures={start=0, end=None})


class LocationSpecification(GT4PyAstNode):
    name: Name
    location_type: str

    @staticmethod
    def template():
        return ast.withitem(
            context_expr=ast.Call(func=ast.Name(id="location"), args=[ast.Name(id=Capture("location_type"))]),
            optional_vars=Capture("name", default=ast.Name(id=UIDGenerator.get_unique_id(prefix="location"))))


# todo: proper cannonicalization (CanBeCanonicalizedTo[Subscript] ?)
class SubscriptSingle(Expr):
    value: Name
    index: str

    @staticmethod
    def template():
        #return ast.Subscript(value=ast.Name(id=Capture("value")), slice=ast.Index(value=ast.Tuple(elts=Capture("indices"))))
        return ast.Subscript(value=Capture("value"),
                             slice=ast.Index(value=ast.Name(id=Capture("index"))))


class SubscriptMultiple(Expr):
    value: Name
    indices: List[Name]

    @staticmethod
    def template():
        return ast.Subscript(value=Capture("value"), slice=ast.Index(value=ast.Tuple(elts=Capture("indices"))))

        #return ast.Subscript(value=ast.Name(id=Capture("value")),
        #                     slice=Fork(
        #                         ast.Index(value=ast.Tuple(elts=Capture("indices", op=ArrayWrap)),
        #                         ast.Index(value=ast.Name(id=Capture("index")))
        #                     )))


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr

    @staticmethod
    def template():
        return ast.BinOp(op=Capture("op"), left=Capture("left"), right=Capture("right"))


class Call(Expr):
    args: List[Expr]
    func: str

    @staticmethod
    def template():
        return ast.Call(args=Capture("args"), func=ast.Name(id=Capture("func")))

class LocationComprehension(GT4PyAstNode):
    target: Name
    iter: Call

    @staticmethod
    def template():
        return ast.comprehension(target=Capture("target"), iter=Capture("iter"))

class Generator(Expr):
    generators: List[LocationComprehension]
    elt: Expr

    @staticmethod
    def template():
        return ast.GeneratorExp(generators=Capture("generators"), elt=Capture("elt"))


#class SubscriptRef(Subscript):
#    pass
#    # todo: validate target is ref


class Assign(Statement):
    target: Name  # todo: allow subscript
    value: Expr

    @staticmethod
    def template():
        return ast.Assign(targets=[Capture("target")], value=Capture("value"))


class Stencil(GT4PyAstNode):
    iteration_spec: List[Union[IterationOrder, LocationSpecification, Interval]]
    body: List[Statement]

    @staticmethod
    def template():
        return ast.With(items=Capture("iteration_spec"), body=Capture("body"))


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


class Argument(GT4PyAstNode):
    name: str
    type: Union[Name, Union[SubscriptMultiple, SubscriptSingle]]
    #is_keyword: bool

    @staticmethod
    def template():
        return ast.arg(arg=Capture("name"), annotation=Capture("type"))

#class Call(Generic[T]):
#    name: str
#    return_type: T
#    arg_types: Ts
#    args: List[Expr]

class Computation(GT4PyAstNode):
    name: str
    arguments: List[Argument]
    stencils: List[Stencil]

    @staticmethod
    def template():
        return ast.FunctionDef(args=ast.arguments(args=Capture("arguments")), body=Capture("stencils"), name=Capture("name"))


def _all_subclasses(cls):
    if inspect.isclass(cls) and issubclass(cls, GT4PyAstNode):
        result = set()
        result.add(cls)
        result.update(cls.__subclasses__())
        result.update([s for c in cls.__subclasses__() for s in _all_subclasses(c) if not inspect.isabstract(c)])
        return result
    elif typing_inspect.is_union_type(cls):
        result = set()
        for el_cls in typing_inspect.get_args(cls):
            result.update(_all_subclasses(el_cls))
        return result
    elif cls in [str, int, type(None)]: # todo: enhance
        return set([cls])

    raise ValueError() # todo: proper error message


def transform_py_ast_into_gt4py_ast(node, eligable_node_types=[Computation]):
    if isinstance(node, ast.AST):
        # visit node fields and transform
        # todo: check if multiple nodes match and throw an error in that case
        # disadvantage: templates can be ambiguous
        for node_type in eligable_node_types:
            if not hasattr(node_type, "template"):
                continue
            captures = {}
            if not anm.match(node, node_type.template(), captures=captures):
                continue
            transformed_captures = {}
            for name, capture in captures.items():
                # determine eligible capture types
                field_type = node_type.__annotations__[name]
                if typing_inspect.get_origin(field_type) == list:
                    el_type = typing_inspect.get_args(field_type)[0]
                    eligible_capture_types = _all_subclasses(el_type)

                    transformed_captures[name] = []
                    for child_capture in capture:
                        transformed_captures[name].append(transform_py_ast_into_gt4py_ast(child_capture, eligible_capture_types))
                else:
                    eligible_capture_types = _all_subclasses(field_type)
                    transformed_captures[name] = transform_py_ast_into_gt4py_ast(capture, eligible_capture_types)
            return node_type(**transformed_captures)
        raise "blub"
    elif type(node) in eligable_node_types:
        return node

    raise ValueError() # todo: improve error

#@gtast(ast.BinOp)
#class BinaryOp(GT4PyAstNode):
#    left: Expr
#    right: Expr
#    op: BinaryOp
