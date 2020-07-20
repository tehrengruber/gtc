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


# from types import MappingProxyType
# from typing import ClassVar, Mapping

import eve  # noqa: F401
from eve.core import NodeTranslator
from gt_toolchain.unstructured import gtir, nir


class GtirToNir(NodeTranslator):
    def visit_NeighborChain(self, node: gtir.NeighborChain, **kwargs):
        return nir.NeighborChain(elements=node.elements)

    def visit_HorizontalDimension(self, node: gtir.HorizontalDimension, **kwargs):
        return nir.HorizontalDimension(
            primary=node.primary, secondary=self.visit(node.secondary) if node.secondary else None
        )

    def visit_VerticalDimension(self, node: gtir.VerticalDimension, **kwargs):
        return nir.VerticalDimension()

    def visit_Dimensions(self, node: gtir.Dimensions, **kwargs):
        return nir.Dimensions(
            horizontal=self.visit(node.horizontal) if node.horizontal else None,
            vertical=self.visit(node.vertical) if node.vertical else None,
        )

    def visit_UField(self, node: gtir.UField, **kwargs):
        return nir.UField(name=node.name, vtype=node.vtype, dimensions=self.visit(node.dimensions))

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs):
        return nir.FieldAccess(name=node.name, location_type=node.location_type)

    def visit_NeighborReduce(self, node: gtir.NeighborReduce, **kwargs):
        return nir.Expr(location_type=node.location_type)  # TODO

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        return nir.BinaryOp(
            left=self.visit(node.left),
            op=node.op,
            right=self.visit(node.right),
            location_type=node.location_type,
        )

    def visit_AssignStmt(self, node: gtir.AssignStmt, **kwargs):
        return nir.AssignStmt(
            left=self.visit(node.left),
            right=self.visit(node.right),
            location_type=node.location_type,
        )

    def visit_HorizontalLoop(self, node: gtir.HorizontalLoop, **kwargs):
        return nir.HorizontalLoop(
            stmt=nir.BlockStmt(
                statements=[self.visit(node.stmt)], location_type=node.stmt.location_type,
            ),
            location_type=node.location_type,
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        return nir.VerticalLoop(
            horizontal_loops=[self.visit(h) for h in node.horizontal_loops],
            loop_order=node.loop_order,
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return nir.Stencil(vertical_loops=[self.visit(loop) for loop in node.vertical_loops])
        # TODO

    def visit_Computation(self, node: gtir.Stencil, **kwargs):
        return nir.Computation(
            name=node.name,
            params=[self.visit(p) for p in node.params],
            stencils=[self.visit(s) for s in node.stencils],
        )
