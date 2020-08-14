# -*- coding: utf-8 -*-
# Eve toolchain

import os

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gtc import common
from gtc.common import LocationType
from gtc.unstructured import ugpu2_codegen
from gtc.unstructured.ugpu2 import (
    AssignStmt,
    BinaryOp,
    Computation,
    Connectivity,
    FieldAccess,
    Kernel,
    Literal,
    NeighborChain,
    NeighborLoop,
    SidComposite,
    SidCompositeEntry,
    Temporary,
    UField,
    KernelCall,
    VerticalDimension,
)


nabla_edge_1_primary_composite = SidComposite(
    name="e",
    location=NeighborChain(elements=[LocationType.Edge]),
    entries=[
        SidCompositeEntry(name="zavg_tmp"),
        SidCompositeEntry(name="zavgS_MXX"),
        SidCompositeEntry(name="zavgS_MYY"),
        SidCompositeEntry(name="S_MXX"),
        SidCompositeEntry(name="S_MYY"),
    ],
)
nabla_vertex_composite = SidComposite(
    name="v_on_e",
    location=NeighborChain(elements=[LocationType.Edge, LocationType.Vertex]),
    entries=[SidCompositeEntry(name="pp")],
)


pp_acc = FieldAccess(name="pp", sid="v_on_e", location_type=LocationType.Vertex)
zavg_tmp_acc = FieldAccess(name="zavg_tmp", sid="e", location_type=LocationType.Edge)
zavg_tmp_acc_on_v = FieldAccess(name="zavg_tmp", sid="e", location_type=LocationType.Vertex)
zavgS_MXX_acc = FieldAccess(name="zavgS_MXX", sid="e", location_type=LocationType.Edge)
zavgS_MYY_acc = FieldAccess(name="zavgS_MYY", sid="e", location_type=LocationType.Edge)
S_MXX_acc = FieldAccess(name="S_MXX", sid="e", location_type=LocationType.Edge)
S_MYY_acc = FieldAccess(name="S_MYY", sid="e", location_type=LocationType.Edge)

edge1_assign0 = AssignStmt(
    left=zavg_tmp_acc,
    right=Literal(value="0.0", location_type=LocationType.Edge, vtype=common.DataType.FLOAT64),
)

edge1_assign1 = AssignStmt(
    left=zavg_tmp_acc,
    right=BinaryOp(
        left=Literal(value="0.5", location_type=LocationType.Edge, vtype=common.DataType.FLOAT64),
        right=zavg_tmp_acc,
        op=common.BinaryOperator.MUL,
    ),
)
edge1_assign2 = AssignStmt(
    left=zavgS_MXX_acc,
    right=BinaryOp(left=S_MXX_acc, right=zavg_tmp_acc, op=common.BinaryOperator.MUL),
)
edge1_assign3 = AssignStmt(
    left=zavgS_MYY_acc,
    right=BinaryOp(left=S_MYY_acc, right=zavg_tmp_acc, op=common.BinaryOperator.MUL),
)

vertex_on_edge_loop = NeighborLoop(
    body_location_type=LocationType.Vertex,
    location_type=LocationType.Edge,
    outer_sid="e",
    sid="v_on_e",
    connectivity="edge_to_vertex_conn",
    body=[
        AssignStmt(
            location_type=LocationType.Vertex,
            left=zavg_tmp_acc_on_v,
            right=BinaryOp(
                left=zavg_tmp_acc_on_v,
                right=pp_acc,
                op=common.BinaryOperator.ADD,
                location_type=LocationType.Vertex,
            ),
        )
    ],
)

nabla_edge_1 = Kernel(
    name="nabla_edge_1",
    connectivities=[
        Connectivity(name="edge_conn", chain=NeighborChain(elements=[LocationType.Edge])),
        Connectivity(
            name="edge_to_vertex_conn",
            chain=NeighborChain(elements=[LocationType.Edge, LocationType.Vertex]),
        ),
    ],
    sids=[nabla_edge_1_primary_composite, nabla_vertex_composite],
    primary_connectivity="edge_conn",
    primary_sid="e",
    ast=[edge1_assign0, vertex_on_edge_loop, edge1_assign1, edge1_assign2, edge1_assign3],
)


# pnabla_MXX_acc = FieldAccess(name="pnabla_MXX", location_type=LocationType.Vertex)
# pnabla_MYY_acc = FieldAccess(name="pnabla_MYY", location_type=LocationType.Vertex)
# sign_acc = FieldAccess(name="sign", location_type=LocationType.Vertex)

# nabla_vertex_2_primary_composite = SidComposite(
#     location_type=LocationType.Vertex,
#     entries=[
#         SidCompositeEntry(name="pnabla_MXX"),
#         SidCompositeEntry(name="pnabla_MYY"),
#         SidCompositeEntry(name="sign"),
#     ],
# )

# nabla_vertex_2_to_edge_composite = SidComposite(
#     location_type=LocationType.Edge,
#     entries=[SidCompositeEntry(name="zavgS_MXX"), SidCompositeEntry(name="zavgS_MYY")],
# )

# edge_on_vertex_loop_x = NeighborLoop(
#     body_location_type=LocationType.Edge,
#     location_type=LocationType.Vertex,
#     body=[
#         AssignStmt(
#             location_type=LocationType.Vertex,
#             left=pnabla_MXX_acc,
#             right=BinaryOp(
#                 left=pnabla_MXX_acc,
#                 right=BinaryOp(
#                     left=zavgS_MXX_acc,
#                     right=sign_acc,
#                     op=common.BinaryOperator.MUL,
#                     location_type=LocationType.Edge,
#                 ),
#                 op=common.BinaryOperator.ADD,
#                 location_type=LocationType.Vertex,
#             ),
#         )
#     ],
# )

# edge_on_vertex_loop_y = NeighborLoop(
#     body_location_type=LocationType.Edge,
#     location_type=LocationType.Vertex,
#     body=[
#         AssignStmt(
#             location_type=LocationType.Vertex,
#             left=pnabla_MYY_acc,
#             right=BinaryOp(
#                 left=pnabla_MYY_acc,
#                 right=BinaryOp(
#                     left=zavgS_MYY_acc,
#                     right=sign_acc,
#                     op=common.BinaryOperator.MUL,
#                     location_type=LocationType.Edge,
#                 ),
#                 op=common.BinaryOperator.ADD,
#                 location_type=LocationType.Vertex,
#             ),
#         )
#     ],
# )

# vertex_2_init_to_zero_x = AssignStmt(
#     left=pnabla_MXX_acc,
#     right=Literal(value="0.0", location_type=LocationType.Vertex, vtype=common.DataType.FLOAT64),
#     location_type=LocationType.Vertex,
# )
# vertex_2_init_to_zero_y = AssignStmt(
#     left=pnabla_MYY_acc,
#     right=Literal(value="0.0", location_type=LocationType.Vertex, vtype=common.DataType.FLOAT64),
#     location_type=LocationType.Vertex,
# )


# nabla_vertex_2 = Kernel(
#     name="nabla_vertex_2",
#     primary_connectivity=LocationType.Vertex,
#     primary_sid_composite=nabla_vertex_2_primary_composite,
#     other_connectivities=[NeighborChain(chain=[LocationType.Vertex, LocationType.Edge])],
#     other_sid_composites=[nabla_vertex_2_to_edge_composite],
#     ast=[
#         vertex_2_init_to_zero_x,
#         edge_on_vertex_loop_x,
#         vertex_2_init_to_zero_y,
#         edge_on_vertex_loop_y,
#     ],
# )

# nabla_vertex_4_composite = SidComposite(
#     location_type=LocationType.Vertex,
#     entries=[
#         SidCompositeEntry(name="vol"),
#         SidCompositeEntry(name="pnabla_MXX"),
#         SidCompositeEntry(name="pnabla_MYY"),
#     ],
# )


# pnabla_MXX_acc = FieldAccess(name="pnabla_MXX", location_type=LocationType.Vertex)
# pnabla_MYY_acc = FieldAccess(name="pnabla_MYY", location_type=LocationType.Vertex)
# vol_acc = FieldAccess(name="vol", location_type=LocationType.Vertex)
# div = BinaryOp(left=pnabla_MXX_acc, right=vol_acc, op=common.BinaryOperator.DIV)
# div2 = BinaryOp(left=pnabla_MYY_acc, right=vol_acc, op=common.BinaryOperator.DIV)
# assign = AssignStmt(left=pnabla_MXX_acc, right=div)
# assign2 = AssignStmt(left=pnabla_MYY_acc, right=div2)

# # debug(assign)

# nabla_vertex_4 = Kernel(
#     name="nabla_vertex_4",
#     primary_connectivity=LocationType.Vertex,
#     primary_sid_composite=nabla_vertex_4_composite,
#     ast=[assign, assign2],
# )

S_MXX = UField(
    name="S_MXX", dimensions=[LocationType.Edge, VerticalDimension()], vtype=common.DataType.FLOAT64
)
S_MYY = UField(
    name="S_MYY", dimensions=[LocationType.Edge, VerticalDimension()], vtype=common.DataType.FLOAT64
)
zavgS_MXX = UField(
    name="zavgS_MXX",
    dimensions=[LocationType.Edge, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)
zavgS_MYY = UField(
    name="zavgS_MYY",
    dimensions=[LocationType.Edge, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)
pp = UField(
    name="pp", dimensions=[LocationType.Vertex, VerticalDimension()], vtype=common.DataType.FLOAT64
)
pnabla_MXX = UField(
    name="pnabla_MXX",
    dimensions=[LocationType.Vertex, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)
pnabla_MYY = UField(
    name="pnabla_MYY",
    dimensions=[LocationType.Vertex, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)
vol = UField(
    name="vol", dimensions=[LocationType.Vertex, VerticalDimension()], vtype=common.DataType.FLOAT64
)
sign = UField(
    name="sign",
    dimensions=[
        LocationType.Vertex,
        NeighborChain(elements=[LocationType.Edge]),
        VerticalDimension(),
    ],
    vtype=common.DataType.FLOAT64,
)


zavg_tmp = Temporary(
    name="zavg_tmp",
    dimensions=[LocationType.Edge, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)

comp = Computation(
    name="nabla",
    parameters=[S_MXX, S_MYY, zavgS_MXX, zavgS_MYY, pp, pnabla_MXX, pnabla_MYY, vol, sign],
    temporaries=[zavg_tmp],
    kernels=[nabla_edge_1],  # , nabla_vertex_2, nabla_vertex_4
    ctrlflow_ast=[KernelCall(name="nabla_edge_1")],
)


debug(comp)

generated_code = ugpu2_codegen.UgpuCodeGenerator.apply(comp)
print(generated_code)

# output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_fvm_nabla_ugpu.hpp"
# with open(output_file, "w+") as output:
#     output.write(generated_code)
