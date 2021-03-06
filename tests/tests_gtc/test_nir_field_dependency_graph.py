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


from gtc.unstructured.nir_passes.field_dependency_graph import generate_dependency_graph

from .nir_utils import make_horizontal_loop_with_copy, make_horizontal_loop_with_init


class TestNIRFieldDependencyGraph:
    def test_assignment_from_literal(self):
        loop, write = make_horizontal_loop_with_init("write")

        result = generate_dependency_graph([loop])

        assert len(result.nodes()) == 1
        assert len(result.edges()) == 0

    def test_single_assignment(self):
        loop, write, read = make_horizontal_loop_with_copy("write0", "input", False)

        result = generate_dependency_graph([loop])

        assert len(result.nodes()) == 1
        assert len(result.edges()) == 0

    def test_dependent_assignment(self):
        loop0, write0 = make_horizontal_loop_with_init("write0")
        loop1, write1, read1 = make_horizontal_loop_with_copy("write1", "write0", False)
        loops = [loop0, loop1]

        result = generate_dependency_graph(loops)

        assert len(result.nodes()) == 2
        assert result.has_edge(write0.id_attr_, write1.id_attr_)
        assert result[write0.id_attr_][write1.id_attr_]["extent"] is False

    def test_dependent_assignment_with_extent(self):
        loop0, write0 = make_horizontal_loop_with_init("write0")
        loop1, write1, read1 = make_horizontal_loop_with_copy("write1", "write0", True)
        loops = [loop0, loop1]

        result = generate_dependency_graph(loops)

        assert len(result.nodes()) == 2
        assert result.has_edge(write0.id_attr_, write1.id_attr_)
        assert result[write0.id_attr_][write1.id_attr_]["extent"] is True
