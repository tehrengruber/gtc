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
from typing import Any, Dict, List, Union

import asdl


python_grammar = asdl.ASDLParser().parse(
    """
-- ASDL's 5 builtin types are:
-- identifier, int, string, object, constant

module Python
{
    mod = Module(stmt* body, type_ignore *type_ignores)
        | Interactive(stmt* body)
        | Expression(expr body)
        | FunctionType(expr* argtypes, expr returns)

        -- not really an actual node but useful in Jython's typesystem.
        | Suite(stmt* body)

    stmt = FunctionDef(identifier name, arguments args,
                       stmt* body, expr* decorator_list, expr? returns,
                       string? type_comment)
          | AsyncFunctionDef(identifier name, arguments args,
                             stmt* body, expr* decorator_list, expr? returns,
                             string? type_comment)

          | ClassDef(identifier name,
             expr* bases,
             keyword* keywords,
             stmt* body,
             expr* decorator_list)
          | Return(expr? value)

          | Delete(expr* targets)
          | Assign(expr* targets, expr value, string? type_comment)
          | AugAssign(expr target, operator op, expr value)
          -- 'simple' indicates that we annotate simple name without parens
          | AnnAssign(expr target, expr annotation, expr? value, int simple)

          -- use 'orelse' because else is a keyword in target languages
          | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | While(expr test, stmt* body, stmt* orelse)
          | If(expr test, stmt* body, stmt* orelse)
          | With(withitem* items, stmt* body, string? type_comment)
          | AsyncWith(withitem* items, stmt* body, string? type_comment)

          | Raise(expr? exc, expr? cause)
          | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | Assert(expr test, expr? msg)

          | Import(alias* names)
          | ImportFrom(identifier? module, alias* names, int? level)

          | Global(identifier* names)
          | Nonlocal(identifier* names)
          | Expr(expr value)
          | Pass | Break | Continue

          -- XXX Jython will be different
          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

          -- BoolOp() can use left & right?
    expr = BoolOp(boolop op, expr* values)
         | NamedExpr(expr target, expr value)
         | BinOp(expr left, operator op, expr right)
         | UnaryOp(unaryop op, expr operand)
         | Lambda(arguments args, expr body)
         | IfExp(expr test, expr body, expr orelse)
         | Dict(expr* keys, expr* values)
         | Set(expr* elts)
         | ListComp(expr elt, comprehension* generators)
         | SetComp(expr elt, comprehension* generators)
         | DictComp(expr key, expr value, comprehension* generators)
         | GeneratorExp(expr elt, comprehension* generators)
         -- the grammar constrains where yield expressions can occur
         | Await(expr value)
         | Yield(expr? value)
         | YieldFrom(expr value)
         -- need sequences for compare to distinguish between
         -- x < 4 < 3 and (x < 4) < 3
         | Compare(expr left, cmpop* ops, expr* comparators)
         | Call(expr func, expr* args, keyword* keywords)
         | FormattedValue(expr value, int? conversion, expr? format_spec)
         | JoinedStr(expr* values)
         | Constant(constant value, string? kind)

         -- the following expression can appear in assignment context
         | Attribute(expr value, identifier attr, expr_context ctx)
         | Subscript(expr value, slice slice, expr_context ctx)
         | Starred(expr value, expr_context ctx)
         | Name(identifier id, expr_context ctx)
         | List(expr* elts, expr_context ctx)
         | Tuple(expr* elts, expr_context ctx)

          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    expr_context = Load | Store | Del | AugLoad | AugStore | Param

    slice = Slice(expr? lower, expr? upper, expr? step)
          | ExtSlice(slice* dims)
          | Index(expr value)

    boolop = And | Or

    operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                 | RShift | BitOr | BitXor | BitAnd | FloorDiv

    unaryop = Invert | Not | UAdd | USub

    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn

    comprehension = (expr target, expr iter, expr* ifs, int is_async)

    excepthandler = ExceptHandler(expr? type, identifier? name, stmt* body)
                    attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                 expr* kw_defaults, arg? kwarg, expr* defaults)

    arg = (identifier arg, expr? annotation, string? type_comment)
           attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    -- keyword arguments supplied to call (NULL identifier for **kwargs)
    keyword = (identifier? arg, expr value)

    -- import name with optional 'as' alias.
    alias = (identifier name, identifier? asname)

    withitem = (expr context_expr, expr? optional_vars)

    type_ignore = TypeIgnore(int lineno, string tag)
}
"""
)


def get_optional_fields(obj):
    assert isinstance(obj, asdl.Product) or isinstance(obj, asdl.Constructor)
    return {field.name: field.opt for field in obj.fields}


# create a dict containing for each python ast node whether it is optional
py_ast_opt_fields = {}
for name, obj in python_grammar.types.items():
    if isinstance(obj, asdl.Product):
        py_ast_opt_fields[name] = get_optional_fields(obj)
    elif isinstance(obj, asdl.Sum):
        for c in obj.types:
            py_ast_opt_fields[c.name] = get_optional_fields(c)
    else:
        raise AssertionError("Invalid grammar. Type must either a product or a sum type.")


class Capture:
    """
    Capture node used to identify and capture nodes in a python ast by :py:func:`match`.

    Example
    -------

    .. code-block: python

        ast.Name(id=Capture("some_name"))
    """

    name: str
    default: Any

    def __init__(self, name, default=None):
        self.name = name
        self.default = default


# just some dummy classes for capturing defaults in lists
class _Placeholder:
    pass


class _PlaceholderList(List):
    pass


class _PlaceholderAst(ast.AST):
    pass


def _get_placeholder_node(pattern_node) -> Union[_Placeholder, _PlaceholderList, _PlaceholderAst]:
    if isinstance(pattern_node, List):
        return _PlaceholderList()
    elif isinstance(pattern_node, ast.AST):
        return _PlaceholderAst()

    return _Placeholder()


def _is_placeholder_for(node, pattern_node) -> bool:
    """
    Is the given node a valid placeholder for the pattern node
    """
    if isinstance(pattern_node, List) and isinstance(node, _PlaceholderList):
        return True
    elif isinstance(pattern_node, ast.AST) and isinstance(node, _PlaceholderAst):
        return True

    return False


def _check_optional(pattern_node, captures=None) -> bool:
    """
    Check if the given pattern node is optional and populate the `captures` dict with the default values stored
    in the `Capture` nodes.
    """
    if captures is None:
        captures = {}

    if isinstance(pattern_node, Capture) and pattern_node.default is not None:
        captures[pattern_node.name] = pattern_node.default
        return True
    elif isinstance(pattern_node, ast.AST):
        return all(_check_optional(child_node) for _, child_node in ast.iter_fields(pattern_node))
    return False


def match(concrete_node, pattern_node, captures=None) -> bool:
    """
    Determine if `concrete_node` matches the `pattern_node` and capture values as specified in the pattern
    node into `captures`

    Example
    -------

    .. code-block: python

        captures = {}
        matches = match(ast.Name(id="some_name"), id=Capture("captured_id"), captures=captures)

        assert(matches)
        assert(captures["captured_id"]=="some_name")

    .. code-block: python

        captures = {}
        matches = anm.match(ast.Name(), ast.Name(id=anm.Capture("id", default="some_name")), captures)

        assert matches
        assert captures["id"] == "some_name"
    """
    if captures is None:
        captures = {}

    if isinstance(pattern_node, Capture):
        captures[pattern_node.name] = concrete_node
        return True
    elif type(concrete_node) != type(pattern_node) and not _is_placeholder_for(
        concrete_node, pattern_node
    ):
        return False
    elif isinstance(pattern_node, ast.AST):
        # iterate over the fields of the concrete- and pattern-node side by side and check if they match
        for fieldname, pattern_val in ast.iter_fields(pattern_node):
            is_opt_in_py_ast = py_ast_opt_fields[type(pattern_node).__name__][fieldname]
            if hasattr(concrete_node, fieldname) and (
                not is_opt_in_py_ast or getattr(concrete_node, fieldname) is not None
            ):
                if not match(getattr(concrete_node, fieldname), pattern_val, captures=captures):
                    return False
            else:
                opt_captures: Dict[str, Any] = {}
                is_opt = _check_optional(pattern_val, opt_captures)
                if is_opt:
                    # if the node is optional populate captures from the default values stored in the pattern node
                    captures.update(opt_captures)
                else:
                    return False
        return True
    elif isinstance(pattern_node, List):
        if not isinstance(concrete_node, List):
            return False

        if len(pattern_node) < len(concrete_node):
            return False
        elif len(pattern_node) > len(concrete_node):
            # insert dummy nodes so that we can still call match on the pattern node and capture the defaults
            concrete_node = [
                concrete_node[i] if i < len(concrete_node) else _get_placeholder_node(cpn)
                for i, cpn in enumerate(pattern_node)
            ]

        return all(
            [match(ccn, cpn, captures=captures) for ccn, cpn in zip(concrete_node, pattern_node)]
        )
    elif concrete_node == pattern_node:
        return True

    return False


# todo: pattern node ast.Name(bla=123) matches ast.Name(id="123") since bla is not an attribute
#  this can lead to errors which are hard to track
