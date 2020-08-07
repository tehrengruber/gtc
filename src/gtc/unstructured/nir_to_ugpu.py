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

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gtc import common
from gtc.unstructured import nir, ugpu


def location_type_from_dimensions(dimensions):
    location_type = [dim for dim in dimensions if isinstance(dim, common.LocationType)]
    if len(location_type) == 1:
        return location_type[0]
    elif len(location_type) == 0:
        return None
    else:
        raise ValueError("Invalid!")


class NirToUgpu(eve.NodeTranslator):
    def __init__(self, *, memo: dict = None, **kwargs):
        super().__init__(memo=memo)
        self.fields = dict()  # poor man symbol table

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
        return ugpu.FieldAccess(
            name=node.name, chain=self.visit(node.chain), location_type=node.location_type
        )

    def visit_VarAccess(self, node: nir.VarAccess, **kwargs):
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
        field_accesses = eve.FindNodes().by_type(nir.FieldAccess, node.stmt)

        primary_sid_composite = ugpu.SidComposite(
            chain=ugpu.NeighborChain(chain=[node.location_type]), entries=[]
        )

        secondary_neighbor_chains = set()
        other_sids = {}
        for acc in field_accesses:
            if len(acc.chain.elements) == 1:
                assert acc.chain.elements[0] == node.location_type
                primary_sid_composite.entries.add(ugpu.SidCompositeEntry(name=acc.name))
            else:
                assert (
                    len(acc.chain.elements) == 2
                )  # TODO cannot deal with more than one level of nesting
                secondary_neighbor_chains.add(acc.chain)
                secondary_loc = acc.chain.elements[
                    -1
                ]  # TODO change if we have more than one level of nesting
                if secondary_loc not in other_sids:
                    other_sids[secondary_loc] = ugpu.SidComposite(
                        chain=ugpu.NeighborChain(chain=[node.location_type, secondary_loc]),
                        entries=[],
                    )
                other_sids[secondary_loc].entries.add(ugpu.SidCompositeEntry(name=acc.name))

        other_connectivities = [self.visit(chain) for chain in secondary_neighbor_chains]
        debug(other_connectivities)

        other_sid_composites = list(other_sids.values())

        return ugpu.Kernel(
            name="kernel" + str(node.id_attr_),
            primary_connectivity=node.location_type,
            primary_sid_composite=primary_sid_composite,
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
            converted_tmp = self.visit(tmp)
            kwargs["temporaries"].append(converted_tmp)
            self.fields[converted_tmp.name] = converted_tmp

        kernels = []
        for loop in node.vertical_loops:
            kernels.extend(self.visit(loop, **kwargs))
        return kernels

    def visit_Computation(self, node: nir.Computation, **kwargs):
        temporaries = []
        parameters = []
        for f in node.params:  # before visiting stencils!
            converted_param = self.visit(f)
            parameters.append(converted_param)
            self.fields[converted_param.name] = converted_param

        kernels = []
        for s in node.stencils:
            kernels += self.visit(s, temporaries=temporaries)

        return ugpu.Computation(
            name=node.name, parameters=parameters, temporaries=temporaries, kernels=kernels,
        )
