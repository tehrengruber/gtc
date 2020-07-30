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


from eve import Bool, Str
from gt_toolchain import common
from gt_toolchain.unstructured import nir
from gt_toolchain.unstructured.nir_passes.field_dependency_graph import generate_dependency_graph


default_vtype = common.DataType.FLOAT32
default_location = common.LocationType.Vertex


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


def make_horizontal_loop_with_assignment(write: Str, read: Str, read_has_extent: Bool):
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


class TestNIRFieldDependencyGraph:
    def test_assignment_from_literal(self):
        loop, write = make_horizontal_loop_with_init("write")

        result = generate_dependency_graph([loop])

        assert len(result.nodes()) == 1
        assert len(result.edges()) == 0

    def test_single_assignment(self):
        loop, write, read = make_horizontal_loop_with_assignment("write0", "input", False)

        result = generate_dependency_graph([loop])

        assert len(result.nodes()) == 1
        assert len(result.edges()) == 0

    def test_dependent_assignment(self):
        loop0, write0 = make_horizontal_loop_with_init("write0")
        loop1, write1, read1 = make_horizontal_loop_with_assignment("write1", "write0", False)
        loops = [loop0, loop1]

        result = generate_dependency_graph(loops)

        assert len(result.nodes()) == 2
        assert result.has_edge(write0.id_attr_, write1.id_attr_)
        assert result[write0.id_attr_][write1.id_attr_]["extent"] is False

    def test_dependent_assignment_with_extent(self):
        loop0, write0 = make_horizontal_loop_with_init("write0")
        loop1, write1, read1 = make_horizontal_loop_with_assignment("write1", "write0", True)
        loops = [loop0, loop1]

        result = generate_dependency_graph(loops)

        assert len(result.nodes()) == 2
        assert result.has_edge(write0.id_attr_, write1.id_attr_)
        assert result[write0.id_attr_][write1.id_attr_]["extent"] is True
