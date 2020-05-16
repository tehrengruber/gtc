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

import pydantic
import pytest

import eve

from .. import common


class TestSourceLocation:
    def test_valid_position(self):
        eve.SourceLocation(line=1, column=1, source="source")

    def test_invalid_position(self):
        with pytest.raises(pydantic.ValidationError):
            eve.SourceLocation(line=1, column=-1, source="source")

    def test_str(self):
        loc = eve.SourceLocation(line=1, column=1, source="source")
        assert str(loc) == "<source: Line 1, Col 1>"


class TestNodeMetaAttributes:
    def test_unique_id(self, sample_node_maker):
        node_a = sample_node_maker()
        node_b = sample_node_maker()
        node_c = sample_node_maker()

        assert node_a.id_attr != node_b.id_attr != node_c.id_attr

    def test_custom_id(self, source_location, sample_node_maker):
        custom_id = "my_custom_id"
        my_node = common.LocationNode(id_attr=custom_id, loc=source_location)
        other_node = sample_node_maker()

        assert my_node.id_attr == custom_id
        assert my_node.id_attr != other_node.id_attr

    # def test_dialect(self, sample_node):
    #     assert sample_node.dialect is sample_node.__class__.dialect
    #     assert issubclass(sample_node.dialect, eve.BaseDialect)
    #     assert sample_node.dialect.name in eve.registered_dialects


# class TestNodeFeatures:
#     def test_validation(self, invalid_sample_node_maker):
#         with pytest.raises(pydantic.ValidationError):
#             invalid_sample_node_maker()

#     def test_mutability(self, mutable_sample_node):
#         mutable_sample_node.id_attr = None

#     def test_inmutability(self, sample_node):
#         with pytest.raises(TypeError):
#             sample_node.id_attr = None

#     def test_attributes(self, sample_node):
#         attribute_names = set(name for name, _ in sample_node.attributes())

#         assert all(name.endswith("_attr") for name in attribute_names)
#         assert not any(name.endswith("_") for name in attribute_names)
#         assert (
#             set(name for name in sample_node.__fields__.keys() if name.endswith("_attr"))
#             == attribute_names
#         )

#     def test_children(self, sample_node):
#         attribute_names = set(name for name, _ in sample_node.attributes())
#         children_names = set(name for name, _ in sample_node.children())
#         public_names = attribute_names | children_names
#         field_names = set(sample_node.__fields__.keys())

#         assert not any(name.endswith("_attr") for name in children_names)
#         assert not any(name.endswith("_") for name in children_names)

#         assert public_names <= field_names
#         assert all(name.endswith("_") for name in field_names - public_names)
