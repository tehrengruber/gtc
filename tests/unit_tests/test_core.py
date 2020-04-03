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

# from devtools import debug
import pydantic
import pytest  # type: ignore

import eve  # type: ignore

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
    def test_unique_id(self, sample_node, sample_node_maker):
        node_a = sample_node_maker()
        node_b = sample_node_maker()
        node_c = sample_node_maker()

        assert node_a.id_attr != node_b.id_attr != node_c.id_attr != sample_node.id_attr

    def test_custom_id(self, source_location, sample_node_maker):
        custom_id = "my_custom_id"
        my_node = common.LocationNode(id_attr=custom_id, loc=source_location)
        other_node = sample_node_maker()

        assert my_node.id_attr == custom_id
        assert my_node.id_attr != other_node.id_attr

    def test_kind(self, sample_node):
        assert sample_node.kind_attr == type(sample_node).__name__

    def test_custom_kind(self, source_location, location_node):
        custom_kind = "LocationNode"
        my_node = common.LocationNode(kind_attr=custom_kind, loc=source_location)

        assert my_node.kind_attr == location_node.kind_attr == custom_kind

        with pytest.raises(pydantic.ValidationError):
            common.LocationNode(kind_attr="WrongKind", loc=source_location)

    def test_dialect(self, sample_node):
        assert sample_node.dialect_attr == "common"

    def test_custom_dialect(self, source_location, location_node):
        custom_dialect = "MyDialect"
        my_node = common.LocationNode(dialect_attr=custom_dialect, loc=source_location)

        assert my_node.dialect_attr == custom_dialect != location_node.dialect_attr


class TestNodeValidation:
    def test_invalid_nodes(self, invalid_sample_node_maker):
        with pytest.raises(pydantic.ValidationError):
            invalid_sample_node_maker()
