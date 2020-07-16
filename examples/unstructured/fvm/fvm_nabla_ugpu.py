# -*- coding: utf-8 -*-
# Eve toolchain

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain import common
from gt_toolchain.unstructured import ugpu_codegen
from gt_toolchain.unstructured.ugpu import (
    AssignStmt,
    BinaryOp,
    Computation,
    FieldAccess,
    Kernel,
    LocationType,
    SidComposite,
    SidCompositeEntry,
    SidTag,
)


nabla_vertex_4_composite = SidComposite(
    name="vertex_things",
    entries=[
        SidCompositeEntry(tag="vol", field="vol"),
        SidCompositeEntry(tag="pnabla_MXX", field="pnabla_MXX"),
    ],
)


pnabla_MXX_acc = FieldAccess(
    tag="pnabla_MXX", location_type=LocationType.Vertex, sid_composite="vertex_things"
)
pnabla_MYY_acc = FieldAccess(
    tag="pnabla_MYY", location_type=LocationType.Vertex, sid_composite="vertex_things"
)
vol_acc = FieldAccess(tag="vol", location_type=LocationType.Vertex, sid_composite="vertex_things")
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


comp = Computation(
    name="fvm_nabla",
    tags=[SidTag(name="vol"), SidTag(name="pnabla_MYY"), SidTag(name="pnabla_MXX")],
    kernels=[nabla_vertex_4],
    ast=[],
)


debug(comp)

print(ugpu_codegen.UgpuCodeGenerator.apply(comp))
