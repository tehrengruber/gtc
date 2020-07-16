# -*- coding: utf-8 -*-
# Eve toolchain

import os

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
    SecondaryLocation,
    SidComposite,
    SidCompositeEntry,
    USid,
    VerticalDimension,
)


nabla_vertex_4_composite = SidComposite(
    name="vertex_things",
    entries=[
        SidCompositeEntry(name="vol"),
        SidCompositeEntry(name="pnabla_MXX"),
        SidCompositeEntry(name="pnabla_MYY"),
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

comp = Computation(
    name="fvm_nabla",
    parameters=[S_MXX, S_MYY, zavgS_MXX, zavgS_MYY, pp, pnabla_MXX, pnabla_MYY, vol, sign],
    # tags=[SidTag(name="vol"), SidTag(name="pnabla_MYY"), SidTag(name="pnabla_MXX")],
    kernels=[nabla_vertex_4],
    # ast=[],
)


debug(comp)

generated_code = ugpu_codegen.UgpuCodeGenerator.apply(comp)
print(generated_code)

output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_fvm_nabla_ugpu.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)
