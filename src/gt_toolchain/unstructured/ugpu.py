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


import enum
from typing import List, Optional, Union

from pydantic import root_validator

from eve import IntEnum, Node, Str
from gt_toolchain import common


@enum.unique
class LocationType(IntEnum):
    Vertex = 0
    Edge = 1
    Cell = 2
    NoLocation = 3


class Expr(Node):
    location_type: LocationType
    pass


class Stmt(Node):
    location_type: LocationType
    pass


class FieldAccess(Expr):
    tag: Str
    sid_composite: Str  # via symbol table
    pass


class AssignStmt(Stmt):
    left: Union[FieldAccess]  # VarAccess
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["left"].location_type != values["right"].location_type:
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        elif values["left"] != values["location_type"]:
            raise ValueError("Location type mismatch")

        return values


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["left"].location_type != values["right"].location_type:
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        elif values["left"] != values["location_type"]:
            raise ValueError("Location type mismatch")

        return values


class SidCompositeEntry(Node):
    tag: Str
    field: Str  # ensure exists via symboltable


class SidComposite(Node):
    name: Str  # symbol table entry (other nodes refer to this composite via this name)
    entries: List[SidCompositeEntry]


class NeighborChain(Node):
    chain: List[LocationType]


class Kernel(Node):
    # location_type: LocationType
    name: Str  # symbol table
    primary_connectivity: LocationType
    other_connectivities: Optional[List[NeighborChain]]
    primary_sid_composite: SidComposite
    other_sid_composites: Optional[List[SidComposite]]
    ast: List[Stmt]


class SidTag(Node):
    name: Str


class Computation(Node):
    name: Str
    tags: List[SidTag]
    kernels: List[Kernel]


# template <class ConnV2E /*Connectivity not needed*/, class VertexOrigins, class VertexStrides>
# __global__ void nabla_vertex_4(ConnV2E v2e,
#     VertexOrigins vertex_origins,
#     VertexStrides vertex_strides) {
#     // vertex loop
#     // for (auto const &t : getVertices(LibTag{}, mesh)) {
#     //     pnabla_MXX(deref(LibTag{}, t), k) =
#     //         pnabla_MXX(deref(LibTag{}, t), k) / vol(deref(LibTag{}, t), k);
#     //     pnabla_MYY(deref(LibTag{}, t), k) =
#     //         pnabla_MYY(deref(LibTag{}, t), k) / vol(deref(LibTag{}, t), k);
#     //   }

#     auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#     if (idx >= gridtools::next::connectivity::size(v2e))
#         return;

#     auto vertex_ptrs = vertex_origins();

#     gridtools::sid::shift(vertex_ptrs, gridtools::device::at_key<vertex>(vertex_strides), idx);

#     *gridtools::device::at_key<pnabla_MXX_tag>(vertex_ptrs) /= *gridtools::device::at_key<vol_tag>(vertex_ptrs);
#     *gridtools::device::at_key<pnabla_MYY_tag>(vertex_ptrs) /= *gridtools::device::at_key<vol_tag>(vertex_ptrs);
# }
