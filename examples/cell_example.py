# for c1 in cells(mesh):
#     field1 = sum(f[c1] * f[c2] for c2 in cells(c1)) # cell2cell
#     field2 = sum(fe[e] for e in edges(c1)) # cell2edge


# with location(cell) as c1:
#     field = sum(f[c1] * f[c2] for c2 in cells(c1))
#     field = sum(f[c1] * f[c3] for c3 in cells(c1))

# c2 and c3 have the same neighbor chain, but refer to different locations
# field = sum(f[c2] * sum(f[c3] for c3 in cells(c1)) for c2 in cells(c1))

# # GTIR:

# class FieldAccess:
#     subscript: LocationRef

# access1 = FieldAccess(subscript=LocationRef(c1))
# access2 = FieldAccess(subscript=LocationRef(c2))

# # NIR

# class FieldAccess:
#     offset: NeighborChain

# access1 = FieldAccess(offset=[cell])
# access2 = FieldAccess(offset=[cell,cell])


# for  in cells(outer_c):
#     field +=
#     field +=

import os

from devtools import debug
from gtc.unstructured import gtir
from gtc.unstructured.gtir import (
    UField,
    Dimensions,
    HorizontalLoop,
    NeighborReduce,
    ReduceOperator,
    HorizontalDimension,
    Stencil,
    VerticalLoop,
    FieldAccess,
    NeighborChain,
    AssignStmt,
    LocationRef,
    LocationComprehension,
    Domain,
    BinaryOp,
)
from gtc.common import DataType, LocationType, LoopOrder, BinaryOperator
from gtc.unstructured.gtir_to_nir import GtirToNir
from gtc.unstructured.nir_to_ugpu import NirToUgpu
from gtc.unstructured.ugpu_codegen import UgpuCodeGenerator

field_in = gtir.UField(
    name="field_in",
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Cell)),
    vtype=DataType.FLOAT64,
)
field_out = gtir.UField(
    name="field_out",
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Cell)),
    vtype=DataType.FLOAT64,
)

# with location(cell) as c1:
#     field1 = sum(f[c1] * f[c2] for c2 in neighbors(c1, Cell, Cell)) # cell2cell

# neighbor_sel = NeighborSelector(
#     neighbor_chain=NeighborChain(elements=[LocationType.Cell, LocationType.Cell])
#     of=LocationRef(name="c1")
# )

# neighbors = NeighborComprehension(
#     location=LocationDecl(name="c2", location_type=LocationType.Cell),
#     neighbors=neighbor_sel
# )
# neighbor_sel = NeighborComprehension(
#     location=LocationDecl(name="c2", location_type=LocationType.Cell),
#     chain=NeighborChain(elements=[LocationType.Cell, LocationType.Cell]),
# )

# neighbor_sel = NeighborSelector(
#      location=LocationDecl(name="c2", chain=NeighborChain(elements=[LocationType.Cell, LocationType.Cell])),
#      of=LocationRef(name="c1")
#  )

# class LocationDecl:


red_operand = BinaryOp(
    op=BinaryOperator.ADD,
    location_type=LocationType.Cell,
    left=FieldAccess(
        name="field_in", location_type=LocationType.Cell, subscript=LocationRef(name="c1")
    ),
    right=FieldAccess(
        name="field_in", location_type=LocationType.Cell, subscript=LocationRef(name="c2")
    ),
)

red_c2c = NeighborReduce(
    op=ReduceOperator.ADD,
    operand=red_operand,
    neighbors=LocationComprehension(
        name="c2",
        chain=NeighborChain(elements=[LocationType.Cell, LocationType.Cell]),
        of=LocationRef(name="c1"),
    ),
    location_type=LocationType.Cell,
)

assign_c2c_red = AssignStmt(
    left=FieldAccess(
        name="field_out", subscript=LocationRef(name="c1"), location_type=LocationType.Cell
    ),
    right=red_c2c,
)


sten = Stencil(
    vertical_loops=[
        VerticalLoop(
            loop_order=LoopOrder.FORWARD,
            horizontal_loops=[
                HorizontalLoop(
                    location=LocationComprehension(
                        name="c1", chain=NeighborChain(elements=[LocationType.Cell]), of=Domain()
                    ),
                    stmt=assign_c2c_red,
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

output_file = os.path.dirname(os.path.realpath(__file__)) + "/cell_example_generated.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)
