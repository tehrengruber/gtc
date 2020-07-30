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

from typing import List

import pytest
from devtools import debug

from eve import Bool, Str
from gt_toolchain import common
from gt_toolchain.unstructured import nir
from gt_toolchain.unstructured.nir_passes.merge_horizontal_loops import _find_merge_candidates

from .util import FindNodes


default_vtype = common.DataType.FLOAT32
default_location = common.LocationType.Vertex


def make_vertical_loop(horizontal_loops=[]):
    return nir.VerticalLoop(horizontal_loops=horizontal_loops, loop_order=common.LoopOrder.FORWARD)


def make_block_stmt(stmts: List[nir.Stmt], declarations=[]):
    return nir.BlockStmt(
        location_type=stmts[0].location_type if len(stmts) > 0 else common.LocationType.Vertex,
        statements=stmts,
        declarations=declarations,
    )


def make_horizontal_loop(block: nir.BlockStmt):
    return nir.HorizontalLoop(stmt=block, location_type=block.location_type)


def make_empty_block_stmt(location_type: common.LocationType):
    return nir.BlockStmt(location_type=location_type, declarations=[], statements=[])


def make_empty_horizontal_loop(location_type: common.LocationType):
    return make_horizontal_loop(make_empty_block_stmt(location_type))


# TODO share with dependency graph
# write = read
def make_horizontal_loop_with_copy(write: Str, read: Str, read_has_extent: Bool):
    loc_type = common.LocationType.Vertex
    write_access = nir.FieldAccess(name=write, extent=False, location_type=loc_type)
    read_access = nir.FieldAccess(name=read, extent=read_has_extent, location_type=loc_type)

    return (
        nir.HorizontalLoop(
            stmt=nir.BlockStmt(
                declarations=[], statements=[nir.AssignStmt(left=write_access, right=read_access)],
            ),
            location_type=loc_type,
        ),
        write_access,
        read_access,
    )


def make_horizontal_loop_with_init(field: Str):
    write_access = nir.FieldAccess(name=field, extent=False, location_type=default_location)
    return (
        nir.HorizontalLoop(
            stmt=nir.BlockStmt(
                declarations=[],
                statements=[
                    nir.AssignStmt(
                        left=write_access,
                        right=nir.Literal(
                            value=common.BuiltInLiteral.ONE,
                            vtype=default_vtype,
                            location_type=default_location,
                        ),
                    )
                ],
            ),
            location_type=default_location,
        ),
        write_access,
    )


class TestNIRMergeHorizontalLoops_NoDependencies:
    def test_same_location(self):
        first_loop = make_empty_horizontal_loop(common.LocationType.Vertex)
        second_loop = make_empty_horizontal_loop(common.LocationType.Vertex)
        stencil = make_vertical_loop([first_loop, second_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 1
        assert result[0][0] == first_loop
        assert result[0][1] == second_loop

    def test_2_on_same_location_1_other(self):
        first_loop = make_empty_horizontal_loop(common.LocationType.Vertex)
        second_loop = make_empty_horizontal_loop(common.LocationType.Vertex)
        third_loop = make_empty_horizontal_loop(common.LocationType.Edge)
        stencil = make_vertical_loop([first_loop, second_loop, third_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0] == first_loop
        assert result[0][1] == second_loop

    def test_2_sets_of_location(self):
        first_loop = make_empty_horizontal_loop(common.LocationType.Vertex)
        second_loop = make_empty_horizontal_loop(common.LocationType.Vertex)
        third_loop = make_empty_horizontal_loop(common.LocationType.Edge)
        fourth_loop = make_empty_horizontal_loop(common.LocationType.Edge)
        stencil = make_vertical_loop([first_loop, second_loop, third_loop, fourth_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 2

        assert len(result[0]) == 2
        assert result[0][0] == first_loop
        assert result[0][1] == second_loop

        assert len(result[1]) == 2
        assert result[1][0] == third_loop
        assert result[1][1] == fourth_loop

    def test_vertex_edge_vertex(self):
        first_loop = make_empty_horizontal_loop(common.LocationType.Vertex)
        second_loop = make_empty_horizontal_loop(common.LocationType.Edge)
        third_loop = make_empty_horizontal_loop(common.LocationType.Vertex)
        stencil = make_vertical_loop([first_loop, second_loop, third_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 0


class TestNIRMergeHorizontalLoops_WithDependencies:
    # field = ...
    # out = field(extent)
    def test_write_read_with_offset(self):
        first_loop, _ = make_horizontal_loop_with_init("field")
        second_loop, _, _ = make_horizontal_loop_with_copy("out", "field", True)
        stencil = make_vertical_loop([first_loop, second_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 0

    # field = input(extent)
    # out = field
    def test_read_a_with_offset_write_b_read_b_no_offset(self):
        first_loop, _, _ = make_horizontal_loop_with_copy("field", "input", True)
        second_loop, _, _ = make_horizontal_loop_with_copy("out", "field", False)
        stencil = make_vertical_loop([first_loop, second_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 1

    # field = ...
    # out = field
    def test_write_read_no_offset(self):
        first_loop, _ = make_horizontal_loop_with_init("field")
        second_loop, _, _ = make_horizontal_loop_with_copy("out", "field", False)
        stencil = make_vertical_loop([first_loop, second_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 1

    # field = ...
    # field2 = field
    # out = field2
    def test_write_read_no_offset_write_read_no_offset(self):
        first_loop, _ = make_horizontal_loop_with_init("field")
        second_loop, _, _ = make_horizontal_loop_with_copy("field2", "field", False)
        third_loop, _, _ = make_horizontal_loop_with_copy("out", "field2", False)
        stencil = make_vertical_loop([first_loop, second_loop, third_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 1

    # field = ...
    # field2 = field
    # out = field2(extent)
    def test_write_read_no_offset_write_read_with_offset(self):
        first_loop, _ = make_horizontal_loop_with_init("field")
        second_loop, _, _ = make_horizontal_loop_with_copy("field2", "field", False)
        third_loop, _, _ = make_horizontal_loop_with_copy("out", "field2", True)
        stencil = make_vertical_loop([first_loop, second_loop, third_loop])

        result = _find_merge_candidates(stencil)

        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0] == first_loop
        assert result[0][1] == second_loop
