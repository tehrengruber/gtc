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


import collections
import itertools
import operator
from enum import Enum

from pydantic import BaseModel, Field, validator


NOTHING = object()

__unique_counter = itertools.count(1)


def unique_id():
    return next(__unique_counter)


# ---- Definitions -----
class StrEnum(str, Enum):
    """Basic :class:`enum.Enum` subclass compatible with string operations."""

    def __str__(self):
        return self.value


class SourceLocation(BaseModel):
    """Source code location (line, column)"""

    line: int = Field(..., description="Line number (starting at 1)", ge=1)
    column: int = Field(..., description="Column number (starting at 1)", ge=1)


class _InmutableConfig:
    allow_mutation = False


class Node(BaseModel):
    """Base node class."""

    node_id_: int = Field(None, description="Unique node identifier")
    node_kind_: str = Field(None, description="Node kind")

    @validator("node_id_", pre=True, always=True)
    def _node_id_validator(cls, v):
        return v or unique_id()

    @validator("node_kind_", pre=True, always=True)
    def _node_kind_validator(cls, v):
        if v and v != cls.__name__:
            raise ValueError(f"node_kind value '{v}' does not match cls.__name__ {cls.__name__}")

        return v or cls.__name__

    @property
    def children(self):
        return {name: getattr(self, name) for name, value in self}

    def __iter__(self):
        return (
            (name, getattr(self, name))
            for name, value in self.__fields__.items()
            if not name.endswith("_")
        )


class InmutableNode(Node):
    """Base inmutable node class."""

    Config = _InmutableConfig


class NodeVisitor:
    """
    Simple node visitor class based on :class:`ast.NodeVisitor`.

    The base class walks the tree and calls a visitor function for every
    node found. This function may return a value which is forwarded by
    the `visit` method. This class is meant to be subclassed, with the
    subclass adding visitor methods.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node. So a `BinOpExpr` node visit function would
    be `visit_BinOpExpr`. If no visitor function exists for a node,
    it tries to get a visitor function for each of its parent classes
    in the order define by the class' `__mro__` attribute. Finally,
    if no visitor function exists for a node or its parents, the
    `generic_visit` visitor is used instead. This behavior can be changed
    by overriding the `visit` method.

    Don't use the `NodeVisitor` if you want to apply changes to nodes during
    traversing. For this a special visitor exists (`NodeTransformer`) that
    allows modifications.
    """

    def visit(self, node: Node, **kwargs):
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(node, **kwargs)

    def generic_visit(self, node: Node, **kwargs):
        items = []
        if isinstance(node, collections.abc.Mapping):
            items = node.items()
        elif isinstance(node, collections.abc.Iterable) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            items = enumerate(node)
        elif isinstance(node, Node):
            items = node
        else:
            pass

        for key, value in items:
            self.visit(value, **kwargs)


class NodeTransformer(NodeVisitor):
    """Simple :class:`NodeVisitor` subclass based on :class:`ast.NodeTransformer` to modify nodes.

    The `NodeTransformer` will walk the tree and use the return value of the
    visitor methods to replace or remove the old node. If the return value of
    the visitor method is ``NOTHING``, the node will be removed from its location,
    otherwise it is replaced with the return value. The return value may also be
    theoriginal node, in which case no replacement takes place.

    Keep in mind that if the node you're operating on has child nodes you must
    either transform the child nodes yourself or call the :meth:`generic_visit`
    method for the node first.

    Usually you use the transformer like this::

       node = YourTransformer().visit(node)
    """

    def generic_visit(self, node: Node, **kwargs):
        items = []
        if isinstance(node, collections.abc.Mapping):
            items = node.items()
            set_op = operator.setitem
            del_op = operator.delitem
        elif isinstance(node, Node):
            items = node
            set_op = setattr
            del_op = delattr
        elif isinstance(node, collections.abc.Iterable) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            index_shift = 0
            for idx, old_value in enumerate(node):
                new_value = self.visit(old_value, **kwargs)
                if new_value is NOTHING:
                    index_shift += 1
                    del node[idx - index_shift]
                else:
                    node[idx - index_shift] = new_value

        for key, old_value in items:
            new_value = self.visit(old_value, **kwargs)
            if new_value is NOTHING:
                del_op(node, key)
            elif new_value != old_value:
                set_op(node, key, new_value)

        return node
