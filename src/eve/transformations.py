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


import collections.abc
import copy

from pydantic import Field, validator

from .core import Node, NodeVisitor, NodeTransformer, NOTHING


class TransformationPass(NodeVisitor):
    @classmethod
    def apply(cls, node: Node, **kwargs):
        return cls(**kwargs).visit(node)

    def __init__(self, *, memo: dict = None, **kwargs):
        assert memo is None or isinstance(memo, dict)
        self.memo = memo or {}

    def generic_visit(self, node: Node, **kwargs):
        if isinstance(node, (Node, collections.abc.Collection)) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            if isinstance(node, Node):
                new_items = {key: self.visit(value, **kwargs) for key, value in node}
                result = node.__class__(
                    node_id_=node.node_id_,
                    node_kind_=node.node_kind_,
                    **{key: value for key, value in new_items.items() if value is not NOTHING},
                )

            elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)):
                # Sequence or set: create a new container instance with the new values
                new_items = [self.visit(value, **kwargs) for value in node]
                result = node.__class__([value for value in new_items if value is not NOTHING])

            elif isinstance(node, collections.abc.Mapping):
                # Mapping: create a new mapping instance with the new values
                new_items = {key: self.visit(value, **kwargs) for key, value in node.items()}
                result = node.__class__(
                    {key: value for key, value in new_items.items() if value is not NOTHING}
                )

        else:
            result = copy.deepcopy(node, memo=self.memo)

        return result


# Alternative implementation: clone and modify the copy in-place
#
# # class TransformationPass(NodeTransformer):
#     @classmethod
#     def apply(cls, node: Node, **kwargs):
#         return cls(**kwargs).visit(node)

#     def __init__(self, *, memo: dict = None, **kwargs):
#         assert memo is None or isinstance(memo, dict)
#         self.memo = memo or {}

#     def visit(self, node: Node, **kwargs):
#         node = copy.deepcopy(node, self.memo)
#         return super().visit(node, **kwargs)
