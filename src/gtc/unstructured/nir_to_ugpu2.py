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
from gtc.unstructured import nir, ugpu2


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
        return ugpu2.NeighborChain(elements=[location for location in node.elements])

    def visit_VerticalDimension(self, node: nir.VerticalDimension, **kwargs):
        return ugpu2.VerticalDimension()

    def visit_UField(self, node: nir.UField, **kwargs):
        return ugpu2.UField(
            name=node.name, vtype=node.vtype, dimensions=self.convert_dimensions(node.dimensions)
        )

    def visit_TemporaryField(self, node: nir.TemporaryField, **kwargs):
        return ugpu2.Temporary(
            name=node.name, vtype=node.vtype, dimensions=self.convert_dimensions(node.dimensions)
        )

    def visit_BinaryOp(self, node: nir.BinaryOp, **kwargs):
        return ugpu2.BinaryOp(
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            op=node.op,
            location_type=node.location_type,
        )

    def visit_Literal(self, node: nir.Literal, **kwargs):
        return ugpu2.Literal(value=node.value, vtype=node.vtype, location_type=node.location_type)

    def visit_NeighborLoop(self, node: nir.NeighborLoop, **kwargs):
        return ugpu2.NeighborLoop(
            outer_sid=kwargs["sids_tbl"][ugpu2.NeighborChain(elements=[node.location_type])].name,
            connectivity=kwargs["conn_tbl"][node.neighbors].name,
            sid=kwargs["sids_tbl"][node.neighbors].name
            if node.neighbors in kwargs["sids_tbl"]
            else None,
            location_type=node.location_type,
            body_location_type=node.neighbors.elements[-1],
            body=self.visit(node.body, **kwargs),
        )

    def visit_FieldAccess(self, node: nir.FieldAccess, **kwargs):
        return ugpu2.FieldAccess(
            name=node.name,
            sid=kwargs["sids_tbl"][self.visit(node.primary, **kwargs)].name,
            location_type=node.location_type,
        )

    def visit_VarAccess(self, node: nir.VarAccess, **kwargs):
        return ugpu2.VarAccess(name=node.name, location_type=node.location_type)

    def visit_AssignStmt(self, node: nir.AssignStmt, **kwargs):
        return ugpu2.AssignStmt(
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            location_type=node.location_type,
        )

    def visit_BlockStmt(self, node: nir.BlockStmt, **kwargs):
        statements = []
        for decl in node.declarations:
            statements.append(
                ugpu2.VarDecl(
                    name=decl.name,
                    init=ugpu2.Literal(
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
        location_type_str = str(common.LocationType(node.location_type).name).lower()
        primary_connectivity = location_type_str + "_conn"
        connectivities = set()
        connectivities.add(
            ugpu2.Connectivity(
                name=primary_connectivity, chain=ugpu2.NeighborChain(elements=[node.location_type])
            )
        )

        field_accesses = eve.FindNodes().by_type(nir.FieldAccess, node.stmt)

        other_sids_entries = {}
        primary_sid_entries = set()
        for acc in field_accesses:
            if len(acc.primary.elements) == 1:
                assert acc.primary.elements[0] == node.location_type
                primary_sid_entries.add(ugpu2.SidCompositeEntry(name=acc.name))
            else:
                assert (
                    len(acc.primary.elements) == 2
                )  # TODO cannot deal with more than one level of nesting
                secondary_loc = acc.primary.elements[
                    -1
                ]  # TODO change if we have more than one level of nesting
                if secondary_loc not in other_sids_entries:
                    other_sids_entries[secondary_loc] = set()
                other_sids_entries[secondary_loc].add(ugpu2.SidCompositeEntry(name=acc.name))

        neighloops = eve.FindNodes().by_type(nir.NeighborLoop, node.stmt)
        for loop in neighloops:
            transformed_neighbors = self.visit(loop.neighbors, **kwargs)
            connectivity_name = str(transformed_neighbors) + "_conn"
            connectivities.add(
                ugpu2.Connectivity(name=connectivity_name, chain=transformed_neighbors)
            )
            primary_sid_entries.add(
                ugpu2.SidCompositeNeighborTableEntry(connectivity=connectivity_name)
            )

        primary_sid = location_type_str
        sids = []
        sids.append(
            ugpu2.SidComposite(
                name=primary_sid,
                entries=primary_sid_entries,
                location=ugpu2.NeighborChain(elements=[node.location_type]),
            )
        )

        for k, v in other_sids_entries.items():
            chain = ugpu2.NeighborChain(elements=[node.location_type, k])
            sids.append(
                ugpu2.SidComposite(name=str(chain), entries=v, location=chain)
            )  # TODO _conn via property

        kernel_name = "kernel_" + node.id_attr_
        kernel = ugpu2.Kernel(
            ast=self.visit(
                node.stmt,
                sids_tbl={s.location: s for s in sids},
                conn_tbl={c.chain: c for c in connectivities},
                **kwargs,
            ),
            name=kernel_name,
            primary_connectivity=primary_connectivity,
            primary_sid=primary_sid,
            connectivities=connectivities,
            sids=sids,
        )
        return kernel, ugpu2.KernelCall(name=kernel_name)

    def visit_VerticalLoop(self, node: nir.VerticalLoop, **kwargs):
        # TODO I am completely ignoring k loops at this point!
        kernels = []
        kernel_calls = []
        for loop in node.horizontal_loops:
            k, c = self.visit(loop, **kwargs)
            kernels.append(k)
            kernel_calls.append(c)
        return kernels, kernel_calls

    def visit_Stencil(self, node: nir.Stencil, **kwargs):
        if "temporaries" not in kwargs:
            # TODO own exception type
            raise ValueError("Internal Error: field `temporaries` is missing")
        for tmp in node.declarations or []:
            converted_tmp = self.visit(tmp)
            kwargs["temporaries"].append(converted_tmp)
            self.fields[converted_tmp.name] = converted_tmp

        kernels = []
        kernel_calls = []
        for loop in node.vertical_loops:
            k, c = self.visit(loop, **kwargs)
            kernels.extend(k)
            kernel_calls.extend(c)
        return kernels, kernel_calls

    def visit_Computation(self, node: nir.Computation, **kwargs):
        temporaries = []
        parameters = []
        for f in node.params:  # before visiting stencils!
            converted_param = self.visit(f)
            parameters.append(converted_param)
            self.fields[converted_param.name] = converted_param

        kernels = []
        ctrlflow_ast = []
        for s in node.stencils:
            kernel, kernel_call = self.visit(s, temporaries=temporaries)
            kernels.extend(kernel)
            ctrlflow_ast.extend(kernel_call)

        debug(kernels)

        return ugpu2.Computation(
            name=node.name,
            parameters=parameters,
            temporaries=temporaries,
            kernels=kernels,
            ctrlflow_ast=ctrlflow_ast,
        )
