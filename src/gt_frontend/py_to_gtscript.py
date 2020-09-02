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
import enum
import inspect
import sys
import typing

import typing_inspect

import eve.types
import gtc.common
from eve import UIDGenerator

from . import ast_node_matcher as anm
from . import gtscript_ast
from .ast_node_matcher import Capture


class PyToGTScript:
    @staticmethod
    def _all_subclasses(typ, *, module=None):
        """
            Return all subclasses of a given type. The type must be one of
             - GT4PyAstNode (returns all subclasses of the given class)
             - Union (return the subclasses of the united)
             - ForwardRef (resolve the reference given the specified module and return its subclasses)
             - built-in python type: str, int, type(None) (return as is)
        """
        if inspect.isclass(typ) and issubclass(typ, gtscript_ast.GTScriptAstNode):
            result = set()
            result.add(typ)
            result.update(typ.__subclasses__())
            result.update(
                [
                    s
                    for c in typ.__subclasses__()
                    for s in PyToGTScript._all_subclasses(c)
                    if not inspect.isabstract(c)
                ]
            )
            return result
        elif inspect.isclass(typ) and typ in [
            gtc.common.AssignmentKind,
            gtc.common.UnaryOperator,
            gtc.common.BinaryOperator,
        ]:
            # note: other types in gtc.common, e.g. gtc.common.DataType are not valid leaf nodes here as they
            #  map to symbols in the gtscript ast and are resolved there
            assert issubclass(typ, enum.Enum)
            return set([typ])
        elif typing_inspect.is_union_type(typ):
            result = set()
            for el_cls in typing_inspect.get_args(typ):
                result.update(PyToGTScript._all_subclasses(el_cls, module=module))
            return result
        elif isinstance(typ, typing.ForwardRef):
            type_name = typing_inspect.get_forward_arg(typ)
            if not hasattr(module, type_name):
                raise ValueError(
                    "Reference to type `{}` in `ForwardRef` not found in module {}".format(
                        type_name, module.__name__
                    )
                )
            return PyToGTScript._all_subclasses(getattr(module, type_name), module=module)
        elif typ in [
            eve.types.StrictStr,
            eve.types.StrictInt,
            eve.types.StrictFloat,
            str,
            int,
            float,
            type(None),
        ]:  # todo: enhance
            return set([typ])

        raise ValueError("Invalid field type {}".format(typ))

    # template is a 1-to-1 mapping from context and python ast node to gt4py ast node. context is encoded in the field types
    # all understood sementic is encoded in the structure
    class Templates:
        Symbol = ast.Name(id=Capture("name"))

        IterationOrder = ast.withitem(
            context_expr=ast.Call(
                func=ast.Name(id="computation"), args=[ast.Name(id=Capture("order"))]
            )
        )

        Constant = ast.Constant(value=Capture("value"))

        Interval = ast.withitem(
            context_expr=ast.Call(
                func=ast.Name(id="interval"), args=[Capture("start"), Capture("stop")]
            )
        )

        LocationSpecification = ast.withitem(
            context_expr=ast.Call(
                func=ast.Name(id="location"), args=[ast.Name(id=Capture("location_type"))]
            ),
            optional_vars=Capture(
                "name", default=ast.Name(id=UIDGenerator.get_unique_id(prefix="location"))
            ),
        )

        SubscriptSingle = ast.Subscript(
            value=Capture("value"), slice=ast.Index(value=ast.Name(id=Capture("index")))
        )

        SubscriptMultiple = ast.Subscript(
            value=Capture("value"), slice=ast.Index(value=ast.Tuple(elts=Capture("indices")))
        )

        BinaryOp = ast.BinOp(op=Capture("op"), left=Capture("left"), right=Capture("right"))

        Call = ast.Call(args=Capture("args"), func=ast.Name(id=Capture("func")))

        LocationComprehension = ast.comprehension(
            target=Capture("target"), iter=Capture("iterator")
        )

        Generator = ast.GeneratorExp(generators=Capture("generators"), elt=Capture("elt"))

        Assign = ast.Assign(targets=[Capture("target")], value=Capture("value"))

        Stencil = ast.With(items=Capture("iteration_spec"), body=Capture("body"))

        Pass = ast.Pass()

        Argument = ast.arg(arg=Capture("name"), annotation=Capture("type_"))

        Computation = ast.FunctionDef(
            args=ast.arguments(args=Capture("arguments")),
            body=Capture("stencils"),
            name=Capture("name"),
        )

    leaf_map = {
        ast.Mult: gtc.common.BinaryOperator.MUL,
        ast.Add: gtc.common.BinaryOperator.ADD,
        ast.Div: gtc.common.BinaryOperator.DIV,
    }

    def transform(self, node, eligible_node_types=None):
        """
        Transform python ast into GTScript ast recursively.
        """
        if eligible_node_types is None:
            eligible_node_types = [gtscript_ast.Computation]

        if isinstance(node, ast.AST):
            is_leaf_node = len(list(ast.iter_fields(node))) == 0
            if is_leaf_node:
                if not type(node) in self.leaf_map:
                    raise ValueError(
                        "Leaf node of type {} found in the python ast can not be mapped."
                    )
                return self.leaf_map[type(node)]
            else:
                # visit node fields and transform
                # todo: check if multiple nodes match and throw an error in that case
                # disadvantage: templates can be ambiguous
                for node_type in eligible_node_types:
                    if not hasattr(self.Templates, node_type.__name__):
                        continue
                    captures = {}
                    if not anm.match(
                        node, getattr(self.Templates, node_type.__name__), captures=captures
                    ):
                        continue
                    module = sys.modules[node_type.__module__]
                    transformed_captures = {}
                    for name, capture in captures.items():
                        assert (
                            name in node_type.__annotations__
                        ), "Invalid capture. No field named `{}` in `{}`".format(
                            name, str(node_type)
                        )
                        field_type = node_type.__annotations__[name]
                        if typing_inspect.get_origin(field_type) == list:
                            # determine eligible capture types
                            el_type = typing_inspect.get_args(field_type)[0]
                            eligible_capture_types = self._all_subclasses(el_type, module=module)

                            # transform captures recursively
                            transformed_captures[name] = []
                            for child_capture in capture:
                                transformed_captures[name].append(
                                    self.transform(child_capture, eligible_capture_types)
                                )
                        else:
                            # determine eligible capture types
                            eligible_capture_types = self._all_subclasses(field_type, module=module)
                            # transform captures recursively
                            transformed_captures[name] = self.transform(
                                capture, eligible_capture_types
                            )
                    return node_type(**transformed_captures)
                raise ValueError(
                    "Expected a node of type {}".format(
                        ", ".join([ent.__name__ for ent in eligible_node_types])
                    )
                )
        elif type(node) in eligible_node_types:
            return node

        raise ValueError(
            "Expected a node of type {}, but got {}".format(
                {*eligible_node_types, ast.AST}, type(node)
            )
        )
