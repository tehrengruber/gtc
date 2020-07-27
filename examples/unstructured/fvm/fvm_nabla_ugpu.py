# -*- coding: utf-8 -*-
# Eve toolchain

import os

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain import common
from gt_toolchain.common import LocationType
from gt_toolchain.unstructured import ugpu_codegen
from gt_toolchain.unstructured.ugpu import (
    AssignStmt,
    BinaryOp,
    Computation,
    FieldAccess,
    Kernel,
    Literal,
    NeighborChain,
    NeighborLoop,
    SecondaryLocation,
    SidComposite,
    SidCompositeEntry,
    Temporary,
    USid,
    VerticalDimension,
)


#         auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#         if (idx >= gridtools::next::connectivity::size(e2v))
#             return;

#         auto edge_ptrs = edge_ptr_holders();

#         gridtools::sid::shift(edge_ptrs, gridtools::device::at_key<edge>(edge_strides), idx);

#         double acc = 0.;
#         { // reduce
#             for (int neigh = 0; neigh < gridtools::next::connectivity::max_neighbors(e2v); ++neigh) {
#                 // body
#                 auto absolute_neigh_index = *gridtools::device::at_key<connectivity_tag>(edge_ptrs);
#                 auto vertex_ptrs = vertex_neighbor_ptr_holders();
#                 gridtools::sid::shift(
#                     vertex_ptrs, gridtools::device::at_key<vertex>(vertex_neighbor_strides), absolute_neigh_index);

#                 acc += *gridtools::device::at_key<pp_tag>(vertex_ptrs);
#                 // body end

#                 gridtools::sid::shift(edge_ptrs, gridtools::device::at_key<neighbor>(edge_strides), 1);
#             }
#             gridtools::sid::shift(edge_ptrs,
#                 gridtools::device::at_key<neighbor>(edge_strides),
#                 -gridtools::next::connectivity::max_neighbors(e2v)); // or reset ptr to origin and shift ?
#         }
#         *gridtools::device::at_key<zavg_tmp_tag>(edge_ptrs) =
#             0.5 * acc; // via temporary for non-optimized parallel model
#         *gridtools::device::at_key<zavgS_MXX_tag>(edge_ptrs) =
#             *gridtools::device::at_key<S_MXX_tag>(edge_ptrs) * *gridtools::device::at_key<zavg_tmp_tag>(edge_ptrs);
#         *gridtools::device::at_key<zavgS_MYY_tag>(edge_ptrs) =
#             *gridtools::device::at_key<S_MYY_tag>(edge_ptrs) * *gridtools::device::at_key<zavg_tmp_tag>(edge_ptrs);
#     }
# }

nabla_edge_1_primary_composite = SidComposite(
    location_type=LocationType.Edge,
    entries=[
        SidCompositeEntry(name="zavg_tmp"),
        SidCompositeEntry(name="zavgS_MXX"),
        SidCompositeEntry(name="zavgS_MYY"),
        SidCompositeEntry(name="S_MXX"),
        SidCompositeEntry(name="S_MYY"),
    ],
)
nabla_vertex_composite = SidComposite(
    location_type=LocationType.Vertex, entries=[SidCompositeEntry(name="pp")],
)


pp_acc = FieldAccess(name="pp", location_type=LocationType.Vertex)
zavg_tmp_acc = FieldAccess(name="zavg_tmp", location_type=LocationType.Edge)
zavgS_MXX_acc = FieldAccess(name="zavgS_MXX", location_type=LocationType.Edge)
zavgS_MYY_acc = FieldAccess(name="zavgS_MYY", location_type=LocationType.Edge)
S_MXX_acc = FieldAccess(name="S_MXX", location_type=LocationType.Edge)
S_MYY_acc = FieldAccess(name="S_MYY", location_type=LocationType.Edge)
# acc_acc = VarAccess(name="acc", location_type=LocationType.Edge)

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
    body=[
        AssignStmt(
            location_type=LocationType.Vertex,
            left=zavg_tmp_acc,
            right=BinaryOp(
                left=zavg_tmp_acc,
                right=pp_acc,
                op=common.BinaryOperator.ADD,
                location_type=LocationType.Vertex,
            ),
        )
    ],
)

nabla_edge_1 = Kernel(
    name="nabla_edge_1",
    primary_connectivity=LocationType.Edge,
    primary_sid_composite=nabla_edge_1_primary_composite,
    other_connectivities=[NeighborChain(chain=[LocationType.Edge, LocationType.Vertex])],
    other_sid_composites=[nabla_vertex_composite],
    ast=[edge1_assign0, vertex_on_edge_loop, edge1_assign1, edge1_assign2, edge1_assign3],
)


# *gridtools::device::at_key<pnabla_MXX_tag>(vertex_ptrs) = 0.;
# { // reduce
#     for (int neigh = 0; neigh < gridtools::next::connectivity::max_neighbors(v2e); ++neigh) {
#         // body
#         auto absolute_neigh_index = *gridtools::device::at_key<connectivity_tag>(vertex_ptrs);
#         if (absolute_neigh_index != gridtools::next::connectivity::skip_value(v2e)) {
#             auto edge_ptrs = edge_neighbor_origins();
#             gridtools::sid::shift(
#                 edge_ptrs, gridtools::device::at_key<edge>(edge_neighbor_strides), absolute_neigh_index);

#             auto zavgS_MXX_value = *gridtools::device::at_key<zavgS_MXX_tag>(edge_ptrs);
#             auto sign_value = *gridtools::device::at_key<sign_tag>(vertex_ptrs);

#             *gridtools::device::at_key<pnabla_MXX_tag>(vertex_ptrs) += zavgS_MXX_value * sign_value;
#             // body end
#         }
#         gridtools::sid::shift(vertex_ptrs, gridtools::device::at_key<neighbor>(vertex_strides), 1);
#     }
#     gridtools::sid::shift(vertex_ptrs,
#         gridtools::device::at_key<neighbor>(vertex_strides),
#         -gridtools::next::connectivity::max_neighbors(v2e)); // or reset ptr to origin and shift ?
# }

pnabla_MXX_acc = FieldAccess(name="pnabla_MXX", location_type=LocationType.Vertex)
pnabla_MYY_acc = FieldAccess(name="pnabla_MYY", location_type=LocationType.Vertex)
# zavgS_MXX_acc = FieldAccess(name="zavgS_MXX", location_type=LocationType.Edge)
# zavgS_MYY_acc = FieldAccess(name="zavgS_MYY", location_type=LocationType.Edge)
sign_acc = FieldAccess(name="sign", location_type=LocationType.Vertex)

nabla_vertex_2_primary_composite = SidComposite(
    location_type=LocationType.Vertex,
    entries=[
        SidCompositeEntry(name="pnabla_MXX"),
        SidCompositeEntry(name="pnabla_MYY"),
        SidCompositeEntry(name="sign"),
    ],
)

nabla_vertex_2_to_edge_composite = SidComposite(
    location_type=LocationType.Edge,
    entries=[SidCompositeEntry(name="zavgS_MXX"), SidCompositeEntry(name="zavgS_MYY")],
)

edge_on_vertex_loop_x = NeighborLoop(
    body_location_type=LocationType.Edge,
    location_type=LocationType.Vertex,
    body=[
        AssignStmt(
            location_type=LocationType.Vertex,
            left=pnabla_MXX_acc,
            right=BinaryOp(
                left=pnabla_MXX_acc,
                right=BinaryOp(
                    left=zavgS_MXX_acc,
                    right=sign_acc,
                    op=common.BinaryOperator.MUL,
                    location_type=LocationType.Edge,
                ),
                op=common.BinaryOperator.ADD,
                location_type=LocationType.Vertex,
            ),
        )
    ],
)

edge_on_vertex_loop_y = NeighborLoop(
    body_location_type=LocationType.Edge,
    location_type=LocationType.Vertex,
    body=[
        AssignStmt(
            location_type=LocationType.Vertex,
            left=pnabla_MYY_acc,
            right=BinaryOp(
                left=pnabla_MYY_acc,
                right=BinaryOp(
                    left=zavgS_MYY_acc,
                    right=sign_acc,
                    op=common.BinaryOperator.MUL,
                    location_type=LocationType.Edge,
                ),
                op=common.BinaryOperator.ADD,
                location_type=LocationType.Vertex,
            ),
        )
    ],
)

vertex_2_init_to_zero_x = AssignStmt(
    left=pnabla_MXX_acc,
    right=Literal(value="0.0", location_type=LocationType.Vertex, vtype=common.DataType.FLOAT64),
    location_type=LocationType.Vertex,
)
vertex_2_init_to_zero_y = AssignStmt(
    left=pnabla_MYY_acc,
    right=Literal(value="0.0", location_type=LocationType.Vertex, vtype=common.DataType.FLOAT64),
    location_type=LocationType.Vertex,
)


nabla_vertex_2 = Kernel(
    name="nabla_vertex_2",
    primary_connectivity=LocationType.Vertex,
    primary_sid_composite=nabla_vertex_2_primary_composite,
    other_connectivities=[NeighborChain(chain=[LocationType.Vertex, LocationType.Edge])],
    other_sid_composites=[nabla_vertex_2_to_edge_composite],
    ast=[
        vertex_2_init_to_zero_x,
        edge_on_vertex_loop_x,
        vertex_2_init_to_zero_y,
        edge_on_vertex_loop_y,
    ],
)

nabla_vertex_4_composite = SidComposite(
    location_type=LocationType.Vertex,
    entries=[
        SidCompositeEntry(name="vol"),
        SidCompositeEntry(name="pnabla_MXX"),
        SidCompositeEntry(name="pnabla_MYY"),
    ],
)


pnabla_MXX_acc = FieldAccess(name="pnabla_MXX", location_type=LocationType.Vertex)
pnabla_MYY_acc = FieldAccess(name="pnabla_MYY", location_type=LocationType.Vertex)
vol_acc = FieldAccess(name="vol", location_type=LocationType.Vertex)
div = BinaryOp(left=pnabla_MXX_acc, right=vol_acc, op=common.BinaryOperator.DIV)
div2 = BinaryOp(left=pnabla_MYY_acc, right=vol_acc, op=common.BinaryOperator.DIV)
assign = AssignStmt(left=pnabla_MXX_acc, right=div)
assign2 = AssignStmt(left=pnabla_MYY_acc, right=div2)

# debug(assign)

nabla_vertex_4 = Kernel(
    name="nabla_vertex_4",
    primary_connectivity=LocationType.Vertex,
    primary_sid_composite=nabla_vertex_4_composite,
    ast=[assign, assign2],
)

S_MXX = USid(name="S_MXX", dimensions=[LocationType.Edge, VerticalDimension()])
S_MYY = USid(name="S_MYY", dimensions=[LocationType.Edge, VerticalDimension()])
zavgS_MXX = USid(name="zavgS_MXX", dimensions=[LocationType.Edge, VerticalDimension()])
zavgS_MYY = USid(name="zavgS_MYY", dimensions=[LocationType.Edge, VerticalDimension()])
pp = USid(name="pp", dimensions=[LocationType.Vertex, VerticalDimension()])
pnabla_MXX = USid(name="pnabla_MXX", dimensions=[LocationType.Vertex, VerticalDimension()])
pnabla_MYY = USid(name="pnabla_MYY", dimensions=[LocationType.Vertex, VerticalDimension()])
vol = USid(name="vol", dimensions=[LocationType.Vertex, VerticalDimension()])
sign = USid(
    name="sign",
    dimensions=[
        LocationType.Vertex,
        SecondaryLocation(chain=[LocationType.Edge]),
        VerticalDimension(),
    ],
)


zavg_tmp = Temporary(name="zavg_tmp", dimensions=[LocationType.Edge, VerticalDimension()])

comp = Computation(
    name="nabla",
    parameters=[S_MXX, S_MYY, zavgS_MXX, zavgS_MYY, pp, pnabla_MXX, pnabla_MYY, vol, sign],
    temporaries=[zavg_tmp],
    # tags=[SidTag(name="vol"), SidTag(name="pnabla_MYY"), SidTag(name="pnabla_MXX")],
    kernels=[nabla_edge_1, nabla_vertex_2, nabla_vertex_4],
    # ast=[],
)


debug(comp)

generated_code = ugpu_codegen.UgpuCodeGenerator.apply(comp)
print(generated_code)

output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_fvm_nabla_ugpu.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)
