# with location(cell) as c1:
#     field1 = sum(f[c1] * f[c2] for c2 in cells(c1)) # cell2cell
#     field2 = sum(fe[e] for e in edges(c1)) # cell2edge


# with location(cell) as c1:
#     field = sum(f[c1] * f[c2] for c2 in cells(c1))
#     field = sum(f[c1] * f[c3] for c3 in cells(c1))

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


from gtc.unstructured import gtir
from gtc.unstructured.gtir import Dimensions, HorizontalDimension
from gtc.common import DataType, LocationType


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


comp = gtir.Computation(params=[field_in, field_out], stencils=[sten])
