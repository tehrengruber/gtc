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

# # --- LLVM-lit test definition
#
# RUN: python -u %s | filecheck %s
#
# CHECK: #define DAWN_GENERATED 1
#
# ---

from gt_toolchain.unstructured import common, naive
from gt_toolchain.unstructured.naive_codegen import NaiveCodeGenerator


field_in = naive.UnstructuredField(
    name="in", location_type=naive.LocationType.Edge, data_type=common.DataType.FLOAT64
)
field_out = naive.UnstructuredField(
    name="out", location_type=naive.LocationType.Node, data_type=common.DataType.FLOAT64
)

zero = naive.LiteralExpr(
    value="0.0", data_type=common.DataType.FLOAT64, location_type=naive.LocationType.Node
)
in_acc = naive.FieldAccessExpr(name="in", offset=[True, 0], location_type=naive.LocationType.Edge)
red = naive.ReduceOverNeighbourExpr(
    operation=common.BinaryOperator.ADD,
    right=in_acc,
    init=zero,
    location_type=naive.LocationType.Node,
)

out_acc = naive.FieldAccessExpr(
    name="out", offset=[False, 0], location_type=naive.LocationType.Node
)

assign = naive.ExprStmt(expr=naive.AssignmentExpr(left=out_acc, right=red))

hori = naive.HorizontalLoop(ast=naive.BlockStmt(statements=[assign]),)
vert = naive.ForK(horizontal_loops=[hori], loop_order=common.LoopOrder.FORWARD)
sten = naive.Stencil(name="reduce_to_node", k_loops=[vert])

comp = naive.Computation(params=[field_in, field_out], stencils=[sten])

generated_code = NaiveCodeGenerator.apply(comp)
print(generated_code)
