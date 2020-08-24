# -*- coding: utf-8 -*-
#
# Simple vertex to edge reduction.
#
# ```python
# for e in edges(mesh):
#     out = sum(in[v] for v in vertices(e))
# ```

import os
import sys

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
from gtc.unstructured.nir_to_usid import NirToUsid
from gtc.unstructured.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


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
)

comp = gtir.Computation(name="sten", params=[field_in, field_out], stencils=[sten])
# debug(comp)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    comp = gtir.Computation(name="sten", params=[field_in, field_out], stencils=[sten])
    nir_comp = GtirToNir().visit(comp)
    debug(nir_comp)
    usid_comp = NirToUsid().visit(nir_comp)
    debug(usid_comp)

    if mode == "unaive":
        generated_code = UsidNaiveCodeGenerator.apply(usid_comp)

    else:  # 'ugpu':
        generated_code = UsidGpuCodeGenerator.apply(usid_comp)

    print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_vertex2edge_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)


if __name__ == "__main__":
    main()
