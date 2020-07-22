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

from typing import Any, Callable, Type

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from eve.core import Node, NodeTranslator, NodeVisitor
from gt_toolchain.unstructured import nir, ugpu


# TODO put in the right place
class FindNodes(NodeVisitor):
    def __init__(self, **kwargs):
        self.result = []

    def visit(self, node: Node, **kwargs) -> Any:
        if kwargs["predicate"](node):
            self.result.append(node)
        self.generic_visit(node, **kwargs)
        return self.result

    @classmethod
    def by_predicate(cls, predicate: Callable[[Node], bool], node: Node, **kwargs):
        return cls().visit(node, predicate=predicate)

    @classmethod
    def by_type(cls, node_type: Type[Node], node: Node, **kwargs):
        def type_predicate(node: Node):
            return isinstance(node, node_type)

        return cls.by_predicate(type_predicate, node)


class NirToUgpu(NodeTranslator):
    def convert_dimensions(self, dims: nir.Dimensions):
        dimensions = []
        if dims.horizontal:
            horizontal = dims.horizontal
            dimensions.append(horizontal.primary)
            if horizontal.secondary:
                dimensions.append(self.visit(horizontal.secondary))
        if dims.vertical:
            dimensions.append(self.visit(dims.vertical))
        return dimensions

    def visit_NeighborChain(self, node: nir.NeighborChain, **kwargs):
        return ugpu.SecondaryLocation(chain=[location for location in node.elements])

    def visit_VerticalDimension(self, node: nir.VerticalDimension, **kwargs):
        return ugpu.VerticalDimension()

    def visit_UField(self, node: nir.UField, **kwargs):
        return ugpu.USid(name=node.name, dimensions=self.convert_dimensions(node.dimensions))

    def visit_TemporaryField(self, node: nir.TemporaryField, **kwargs):
        return ugpu.Temporary(name=node.name, dimensions=self.convert_dimensions(node.dimensions))

    def visit_BinaryOp(self, node: nir.BinaryOp, **kwargs):
        return ugpu.BinaryOp(
            left=self.visit(node.left),
            right=self.visit(node.right),
            op=node.op,
            location_type=node.location_type,
        )

    def visit_Literal(self, node: nir.Literal, **kwargs):
        return ugpu.Literal(value=node.value, vtype=node.vtype, location_type=node.location_type)

    def visit_NeighborLoop(self, node: nir.NeighborLoop, **kwargs):
        return ugpu.NeighborLoop(
            location_type=node.location_type,
            body_location_type=node.neighbors.elements[-1],
            body=self.visit(node.body, **kwargs),
        )

    def visit_FieldAccess(self, node: nir.FieldAccess, **kwargs):
        return ugpu.FieldAccess(name=node.name, location_type=node.location_type)

    def visit_VarAccess(self, node: nir.FieldAccess, **kwargs):
        return ugpu.VarAccess(name=node.name, location_type=node.location_type)

    def visit_AssignStmt(self, node: nir.AssignStmt, **kwargs):
        return ugpu.AssignStmt(
            left=self.visit(node.left),
            right=self.visit(node.right),
            location_type=node.location_type,
        )

    def visit_BlockStmt(self, node: nir.BlockStmt, **kwargs):
        statements = []
        for decl in node.declarations:
            statements.append(
                ugpu.VarDecl(
                    name=decl.name,
                    init=ugpu.Literal(
                        value="0.0", vtype=decl.vtype, location_type=node.location_type
                    ),
                    vtype=decl.vtype,
                    location_type=node.location_type,
                )
            )
        for stmt in node.statements:
            statements.append(self.visit(stmt, **kwargs))
        return statements

    def visit_HorizontalLoop(self, node: nir.HorizontalLoop, **kwargs):
        field_accesses = FindNodes().by_type(nir.FieldAccess, node.stmt)
        secondary_location_types = set([acc.location_type for acc in field_accesses])
        secondary_location_types.remove(node.location_type)
        primary_fields = [
            ugpu.SidCompositeEntry(name=f.name)
            for f in filter(lambda f: f.location_type == node.location_type, field_accesses)
        ]
        other_connectivities = [
            ugpu.NeighborChain(chain=[node.location_type, location])
            for location in secondary_location_types
        ]

        other_sid_composites = []
        for loc in secondary_location_types:
            other_sid_composites.append(
                ugpu.SidComposite(
                    location_type=loc,
                    entries=[
                        ugpu.SidCompositeEntry(name=f.name)
                        for f in filter(lambda f: f.location_type == loc, field_accesses)
                    ],
                )
            )
        return ugpu.Kernel(
            name="kernel" + str(node.id_attr),
            primary_connectivity=node.location_type,
            primary_sid_composite=ugpu.SidComposite(
                location_type=node.location_type,
                entries=primary_fields
                # secondary TODO
            ),
            other_connectivities=other_connectivities,
            other_sid_composites=other_sid_composites,
            ast=self.visit(node.stmt),
        )

    def visit_VerticalLoop(self, node: nir.VerticalLoop, **kwargs):
        # TODO I am completely ignoring k loops at this point!
        return [self.visit(loop, **kwargs) for loop in node.horizontal_loops]

    def visit_Stencil(self, node: nir.Stencil, **kwargs):
        if "temporaries" not in kwargs:
            # TODO own exception type
            raise ValueError("Internal Error: field `temporaries` is missing")
        for tmp in node.declarations or []:
            kwargs["temporaries"].append(self.visit(tmp))
        kernels = []
        for loop in node.vertical_loops:
            kernels.extend(self.visit(loop, **kwargs))
        return kernels

    def visit_Computation(self, node: nir.Computation, **kwargs):
        temporaries = []
        kernels = []
        for s in node.stencils:
            kernels += self.visit(s, temporaries=temporaries)

        return ugpu.Computation(
            name=node.name,
            parameters=[self.visit(f) for f in node.params],
            temporaries=temporaries,
            kernels=kernels,
        )
