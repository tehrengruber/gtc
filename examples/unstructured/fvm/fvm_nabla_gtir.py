# -*- coding: utf-8 -*-
# Eve toolchain

import os

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain.common import BinaryOperator, DataType, LocationType, LoopOrder
from gt_toolchain.unstructured.gtir import (
    AssignStmt,
    BinaryOp,
    Computation,
    Dimensions,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    Literal,
    NeighborChain,
    NeighborReduce,
    ReduceOperator,
    Stencil,
    TemporaryField,
    UField,
    VerticalLoop,
)
from gt_toolchain.unstructured.gtir_to_nir import GtirToNir
from gt_toolchain.unstructured.nir_to_ugpu import NirToUgpu
from gt_toolchain.unstructured.ugpu_codegen import UgpuCodeGenerator


vertical_loops = []
fields = []

S_MXX = UField(
    name="S_MXX",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
)
fields.append(S_MXX)
S_MYY = UField(
    name="S_MYY",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
)
fields.append(S_MYY)

zavgS_MXX = UField(
    name="zavgS_MXX",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
)
fields.append(zavgS_MXX)
zavgS_MYY = UField(
    name="zavgS_MYY",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
)
fields.append(zavgS_MYY)

pp = UField(
    name="pp",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
)
fields.append(pp)

pnabla_MXX = UField(
    name="pnabla_MXX",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
)
fields.append(pnabla_MXX)
pnabla_MYY = UField(
    name="pnabla_MYY",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
)
fields.append(pnabla_MYY)

vol = UField(
    name="vol",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
)
fields.append(vol)

sign = UField(
    name="sign",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(
        horizontal=HorizontalDimension(
            primary=LocationType.Vertex,
            secondary=NeighborChain(elements=[LocationType.Vertex, LocationType.Edge]),
        )
    ),
)
fields.append(sign)

# ===========================

zavg_red = NeighborReduce(
    op=ReduceOperator.ADD,
    operand=FieldAccess(name="pp", location_type=LocationType.Vertex),
    neighbors=NeighborChain(elements=[LocationType.Edge, LocationType.Vertex]),
    location_type=LocationType.Edge,
)
zavg_mul = BinaryOp(
    left=Literal(value="0.5", vtype=DataType.FLOAT64, location_type=LocationType.Edge),
    op=BinaryOperator.MUL,
    right=zavg_red,
)
zavg_assign = AssignStmt(
    left=FieldAccess(name="zavg_tmp", location_type=LocationType.Edge), right=zavg_mul
)

assign_zavgS_MXX = AssignStmt(
    left=FieldAccess(name="zavgS_MXX", location_type=LocationType.Edge),
    right=BinaryOp(
        left=FieldAccess(name="zavg_tmp", location_type=LocationType.Edge),
        op=BinaryOperator.MUL,
        right=FieldAccess(name="S_MXX", location_type=LocationType.Edge),
    ),
)

assign_zavgS_MYY = AssignStmt(
    left=FieldAccess(name="zavgS_MYY", location_type=LocationType.Edge),
    right=BinaryOp(
        left=FieldAccess(name="zavg_tmp", location_type=LocationType.Edge),
        op=BinaryOperator.MUL,
        right=FieldAccess(name="S_MYY", location_type=LocationType.Edge),
    ),
)

vertical_loops.append(
    VerticalLoop(
        loop_order=LoopOrder.FORWARD,
        horizontal_loops=[
            HorizontalLoop(location_type=LocationType.Edge, stmt=zavg_assign),
            HorizontalLoop(location_type=LocationType.Edge, stmt=assign_zavgS_MXX),
            HorizontalLoop(location_type=LocationType.Edge, stmt=assign_zavgS_MYY),
        ],
    )
)

# ===========================


assign_pnabla_MXX = AssignStmt(
    left=FieldAccess(name="pnabla_MXX", location_type=LocationType.Vertex),
    right=NeighborReduce(
        operand=BinaryOp(
            left=FieldAccess(name="zavgS_MXX", location_type=LocationType.Edge),
            op=BinaryOperator.MUL,
            right=FieldAccess(name="sign", location_type=LocationType.Edge),
        ),
        op=ReduceOperator.ADD,
        location_type=LocationType.Vertex,
        neighbors=NeighborChain(elements=[LocationType.Vertex, LocationType.Edge]),
    ),
    location_type=LocationType.Vertex,
)
assign_pnabla_MYY = AssignStmt(
    left=FieldAccess(name="pnabla_MYY", location_type=LocationType.Vertex),
    right=NeighborReduce(
        operand=BinaryOp(
            left=FieldAccess(name="zavgS_MYY", location_type=LocationType.Edge),
            op=BinaryOperator.MUL,
            right=FieldAccess(name="sign", location_type=LocationType.Edge),
        ),
        op=ReduceOperator.ADD,
        location_type=LocationType.Vertex,
        neighbors=NeighborChain(elements=[LocationType.Vertex, LocationType.Edge]),
    ),
    location_type=LocationType.Vertex,
)


vertical_loops.append(
    VerticalLoop(
        loop_order=LoopOrder.FORWARD,
        horizontal_loops=[
            HorizontalLoop(location_type=LocationType.Vertex, stmt=assign_pnabla_MXX),
            HorizontalLoop(location_type=LocationType.Vertex, stmt=assign_pnabla_MYY),
        ],
    )
)

# ===========================
# TODO pole correction for pnabla_MYY
# ===========================

assign_pnabla_MXX_vol = AssignStmt(
    left=FieldAccess(name="pnabla_MXX", location_type=LocationType.Vertex),
    right=BinaryOp(
        left=FieldAccess(name="pnabla_MXX", location_type=LocationType.Vertex),
        op=BinaryOperator.DIV,
        right=FieldAccess(name="vol", location_type=LocationType.Vertex),
    ),
)
assign_pnabla_MYY_vol = AssignStmt(
    left=FieldAccess(name="pnabla_MYY", location_type=LocationType.Vertex),
    right=BinaryOp(
        left=FieldAccess(name="pnabla_MYY", location_type=LocationType.Vertex),
        op=BinaryOperator.DIV,
        right=FieldAccess(name="vol", location_type=LocationType.Vertex),
    ),
)

vertical_loops.append(
    VerticalLoop(
        loop_order=LoopOrder.FORWARD,
        horizontal_loops=[
            HorizontalLoop(location_type=LocationType.Vertex, stmt=assign_pnabla_MXX_vol),
            HorizontalLoop(location_type=LocationType.Vertex, stmt=assign_pnabla_MYY_vol),
        ],
    )
)

nabla_stencil = Stencil(
    vertical_loops=vertical_loops,
    declarations=[
        TemporaryField(
            name="zavg_tmp",
            vtype=DataType.FLOAT64,
            dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
        )
    ],
)

comp = Computation(name="nabla", params=fields, stencils=[nabla_stencil])

nir_comp = GtirToNir().visit(comp)
# debug(nir_comp)
ugpu_comp = NirToUgpu().visit(nir_comp)
# debug(ugpu_comp)

generated_code = UgpuCodeGenerator.apply(ugpu_comp)
print(generated_code)

output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_fvm_nabla_ugpu.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)
