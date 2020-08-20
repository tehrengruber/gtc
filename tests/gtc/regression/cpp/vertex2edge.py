# -*- coding: utf-8 -*-
#
# Simple vertex to edge reduction.
#
# ```python
# for e in edges(mesh):
#     out = sum(in[v] for v in vertices(e))
# ```

import os

from devtools import debug

from gtc.common import DataType, LocationType, LoopOrder
from gtc.unstructured import gtir
from gtc.unstructured.gtir import (
    AssignStmt,
    Dimensions,
    Domain,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    LocationComprehension,
    LocationRef,
    NeighborChain,
    NeighborReduce,
    ReduceOperator,
    Stencil,
    UField,
    VerticalLoop,
)
from gtc.unstructured.gtir_to_nir import GtirToNir
from gtc.unstructured.nir_to_ugpu import NirToUgpu
from gtc.unstructured.ugpu_codegen import UgpuCodeGenerator
from gtc.unstructured.unaive_codegen import UnaiveCodeGenerator


field_in = UField(
    name="field_in",
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
    vtype=DataType.FLOAT64,
)
field_out = UField(
    name="field_out",
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
    vtype=DataType.FLOAT64,
)

red_operand = FieldAccess(
    name="field_in", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
)


red_v2e = NeighborReduce(
    op=ReduceOperator.ADD,
    operand=red_operand,
    neighbors=LocationComprehension(
        name="v",
        chain=NeighborChain(elements=[LocationType.Edge, LocationType.Vertex]),
        of=LocationRef(name="e"),
    ),
    location_type=LocationType.Edge,
)

assign_v2e_red = AssignStmt(
    left=FieldAccess(
        name="field_out", subscript=[LocationRef(name="e")], location_type=LocationType.Edge
    ),
    right=red_v2e,
)


sten = Stencil(
    vertical_loops=[
        VerticalLoop(
            loop_order=LoopOrder.FORWARD,
            horizontal_loops=[
                HorizontalLoop(
                    location=LocationComprehension(
                        name="e", chain=NeighborChain(elements=[LocationType.Edge]), of=Domain()
                    ),
                    stmt=assign_v2e_red,
                )
            ],
        )
    ],
    declarations=[],
)

comp = gtir.Computation(name="sten", params=[field_in, field_out], stencils=[sten])
# debug(comp)

nir_comp = GtirToNir().visit(comp)
debug(nir_comp)
ugpu_comp = NirToUgpu().visit(nir_comp)
debug(ugpu_comp)

generated_code = UgpuCodeGenerator.apply(ugpu_comp)
print(generated_code)

output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_vertex2edge_ugpu.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)

generated_code = UnaiveCodeGenerator.apply(ugpu_comp)
output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_vertex2edge_unaive.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)
