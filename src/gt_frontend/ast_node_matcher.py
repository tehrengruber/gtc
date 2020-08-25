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
from typing import Any, Dict, List, Tuple


class Capture:
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


def _get_placeholder_node(pattern_node):
    if isinstance(pattern_node, List):
        return _PlaceholderList()
    elif isinstance(pattern_node, ast.AST):
        return _PlaceholderAst()

    return _Placeholder()


def _is_placeholder_for(node, pattern_node):
    """
    Is the given node a valid placeholder for the pattern node
    """
    if isinstance(pattern_node, List) and isinstance(node, _PlaceholderList):
        return True
    elif isinstance(pattern_node, ast.AST) and isinstance(node, _PlaceholderAst):
        return True

    return False


def check_optional(pattern_node, captures=None):
    """
    Check if the given pattern node is optional and populate the `captures` dict with the default values stored
    in the `Capture` nodes
    """
    if captures is None:
        captures = {}

    if isinstance(pattern_node, Capture) and pattern_node.default is not None:
        captures[pattern_node.name] = pattern_node.default
        return True
    elif isinstance(pattern_node, ast.AST):
        is_optional = True
        for _fieldname, child_node in ast.iter_fields(pattern_node):
            is_optional &= check_optional(child_node, captures)
        return is_optional
    return False


def match(concrete_node, pattern_node, captures=None) -> Tuple[bool, Dict[str, ast.AST]]:
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
            if hasattr(concrete_node, fieldname):
                if not match(getattr(concrete_node, fieldname), pattern_val, captures=captures):
                    return False
            else:
                opt_captures = {}
                is_opt = check_optional(pattern_val, opt_captures)
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
