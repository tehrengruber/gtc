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


"""Global configuration of test generation and execution with pytest."""

import pytest  # type: ignore

from . import common


# -- Fixtures --
NODE_MAKERS = [
    value
    for key, value in common.__dict__.items()
    if "node" in key and "invalid" not in key and "mutable" not in key and key.endswith("_maker")
]

MUTABLE_NODE_MAKERS = [
    value
    for key, value in common.__dict__.items()
    if "node" in key
    and "invalid" not in key
    and key.startswith("mutable")
    and key.endswith("_maker")
]

INVALID_NODE_MAKERS = [
    value
    for key, value in common.__dict__.items()
    if "node" in key and key.startswith("invalid") and key.endswith("_maker")
]


@pytest.fixture
def source_location_maker():
    return common.source_location_maker


@pytest.fixture
def source_location(source_location_maker):
    return source_location_maker()


@pytest.fixture
def location_node_maker():
    return common.location_node_maker


@pytest.fixture
def location_node(location_node_maker):
    return location_node_maker()


@pytest.fixture
def simple_node_maker():
    return common.simple_node_maker


@pytest.fixture
def simple_node(simple_node_maker):
    return simple_node_maker()


@pytest.fixture
def simple_node_with_optionals_maker():
    return common.simple_node_with_optionals_maker


@pytest.fixture
def simple_node_with_optionals(simple_node_with_optionals_maker):
    return simple_node_with_optionals_maker()


@pytest.fixture
def simple_node_with_hidden_members_maker():
    return common.simple_node_with_hidden_members_maker


@pytest.fixture
def simple_node_with_hidden_members(simple_node_with_hidden_members_maker):
    return simple_node_with_hidden_members_maker()


@pytest.fixture
def simple_node_with_loc_maker():
    return common.simple_node_with_loc_maker


@pytest.fixture
def simple_node_with_loc(simple_node_with_loc_maker):
    return simple_node_with_loc_maker()


@pytest.fixture
def simple_node_with_collections_maker():
    return common.simple_node_with_collections_maker


@pytest.fixture
def simple_node_with_collections(simple_node_with_collections_maker):
    return simple_node_with_collections_maker()


@pytest.fixture
def simple_node_with_abstractcollections_maker():
    return common.simple_node_with_abstractcollections_maker


@pytest.fixture
def simple_node_with_abstractcollections(simple_node_with_abstractcollections_maker):
    return simple_node_with_abstractcollections_maker()


@pytest.fixture
def compound_node_maker():
    return common.compound_node_maker


@pytest.fixture
def compound_node(compound_node_maker):
    return compound_node_maker()


@pytest.fixture
def compound_node_with_collections_maker():
    return common.compound_node_with_collections_maker


@pytest.fixture
def compound_node_with_collections(compound_node_with_collections_maker):
    return compound_node_with_collections_maker()


@pytest.fixture
def mutable_simple_node_maker():
    return common.mutable_simple_node_maker


@pytest.fixture
def mutable_simple_node(mutable_simple_node_maker):
    return mutable_simple_node_maker()


@pytest.fixture
def mutable_compound_node_maker():
    return common.mutable_compound_node_maker


@pytest.fixture
def mutable_compound_node(mutable_compound_node_maker):
    return mutable_compound_node_maker()


@pytest.fixture
def invalid_location_node_maker():
    return common.invalid_location_node_maker()


@pytest.fixture
def invalid_at_int_simple_node_maker():
    return common.invalid_at_int_simple_node_maker()


@pytest.fixture
def invalid_at_float_simple_node_maker():
    return common.invalid_at_float_simple_node_maker()


@pytest.fixture
def invalid_at_str_simple_node_maker():
    return common.invalid_at_str_simple_node_maker()


@pytest.fixture
def invalid_at_bytes_simple_node_maker():
    return common.invalid_at_bytes_simple_node_maker()


@pytest.fixture
def invalid_at_enum_simple_node_maker():
    return common.invalid_at_enum_simple_node_maker()


@pytest.fixture(params=NODE_MAKERS)
def sample_node_maker(request):
    return request.param


@pytest.fixture(params=NODE_MAKERS)
def sample_node(request):
    return request.param()


@pytest.fixture(params=MUTABLE_NODE_MAKERS)
def mutable_sample_node_maker(request):
    return request.param


@pytest.fixture(params=MUTABLE_NODE_MAKERS)
def mutable_sample_node(request):
    return request.param()


@pytest.fixture(params=INVALID_NODE_MAKERS)
def invalid_sample_node_maker(request):
    return request.param
