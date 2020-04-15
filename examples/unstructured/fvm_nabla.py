# -*- coding: utf-8 -*-
# Eve toolchain

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain.unstructured import common, naive, sir, sir_to_naive


# fields = [
#     sir_utils.make_field("S_MXX", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Edge')], 1)),
#     sir_utils.make_field("S_MYY", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Edge')], 1)),
#     sir_utils.make_field("zavgS_MXX", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Edge')], 1)),
#     sir_utils.make_field("zavgS_MYY", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Edge')], 1)),
#     sir_utils.make_field("pp", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex')], 1)),
#     sir_utils.make_field("pnabla_MXX", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex')], 1)),
#     sir_utils.make_field("pnabla_MYY", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex')], 1)),
#     sir_utils.make_field("vol", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex')], 1)),
#     sir_utils.make_field("sign", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex'), SIR.LocationType.Value('Edge')], 1)),
# ]

S_MXX = sir.Field(
    name="S_MXX",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Edge)
    ),
)
S_MYY = sir.Field(
    name="S_MYY",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Edge)
    ),
)
zavgS_MXX = sir.Field(
    name="zavgS_MXX",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Edge)
    ),
)
S_MYY = sir.Field(
    name="S_MYY",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Edge)
    ),
)
pp = sir.Field(
    name="pp",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Vertex)
    ),
)
pnabla_MXX = sir.Field(
    name="pnabla_MXX",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Vertex)
    ),
)
pnabla_MYY = sir.Field(
    name="pnabla_MYY",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Vertex)
    ),
)
sign = sir.Field(
    name="sign",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(
            dense_location_type=sir.LocationType.Vertex,
            sparse_part=[sir.LocationType.Vertex, sir.LocationType.Edge],
        )
    ),
)

# body_ast = sir_utils.make_ast(
#         [
#             sir_utils.make_var_decl_stmt(
#                 sir_utils.make_type(SIR.BuiltinType.Float),
#                 "zavg",
#                 0,
#                 "=",
#                 sir_utils.make_binary_operator(sir_utils.make_literal_access_expr(
#                     "0.5", SIR.BuiltinType.Float), "*", sir_utils.make_reduction_over_neighbor_expr(
#                     "+",
#                     sir_utils.make_field_access_expr("pp"),
#                     sir_utils.make_literal_access_expr(
#                         "0.0", SIR.BuiltinType.Float),
#                     lhs_location=SIR.LocationType.Value('Edge'),
#                     rhs_location=SIR.LocationType.Value('Vertex')
#                     # TODO assumed iflip==0, i.e. current implementation zbc = 1
#                 ))
#             ),
zavg_red = sir.ReductionOverNeighborExpr(
    op="+",
    rhs=sir.FieldAccessExpr(name="pp", vertical_offset=0, horizontal_offset=sir.ZeroOffset()),
    init=sir.LiteralAccessExpr(
        value="0.0", data_type=sir.BuiltinType(type_id=common.DataType.FLOAT32),
    ),
    chain=[sir.LocationType.Edge, sir.LocationType.Vertex],
)
zavg_mul = sir.BinaryOperator(
    op="*",
    left=sir.LiteralAccessExpr(
        value="0.5", data_type=sir.BuiltinType(type_id=common.DataType.FLOAT32),
    ),
    right=zavg_red,
)
zavg_decl = sir.VarDeclStmt(
    data_type=sir.Type(
        data_type=sir.BuiltinType(type_id=common.DataType.FLOAT32),
        is_const=False,
        is_volatile=False,
    ),
    name="zavg",
    dimension=0,
    op="=",
    init_list=[zavg_mul],
)
s2n = sir_to_naive.SirToNaive()
s2n.sir_stencil_params["pp"] = pp
s2n.current_loc_type_stack.append(naive.LocationType.Edge)
debug(s2n.visit(zavg_decl))

#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "zavgS_MXX"), sir_utils.make_binary_operator(sir_utils.make_field_access_expr("S_MXX"), "*", sir_utils.make_var_access_expr("zavg"))),
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "zavgS_MYY"), sir_utils.make_binary_operator(sir_utils.make_field_access_expr("S_MYY"), "*", sir_utils.make_var_access_expr("zavg"))),
#             # ===========================
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "pnabla_MXX"), sir_utils.make_reduction_over_neighbor_expr(
#                     "+",
#                     sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
#                         "zavgS_MXX"), "*", sir_utils.make_field_access_expr("sign")),
#                     sir_utils.make_literal_access_expr(
#                         "0.0", SIR.BuiltinType.Float),
#                     lhs_location=SIR.LocationType.Value('Vertex'),
#                     rhs_location=SIR.LocationType.Value('Edge')
#             )),
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "pnabla_MYY"), sir_utils.make_reduction_over_neighbor_expr(
#                     "+",
#                     sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
#                         "zavgS_MYY"), "*", sir_utils.make_field_access_expr("sign")),
#                     sir_utils.make_literal_access_expr(
#                         "0.0", SIR.BuiltinType.Float),
#                     lhs_location=SIR.LocationType.Value('Vertex'),
#                     rhs_location=SIR.LocationType.Value('Edge')
#             )),
#             # ===========================
#             # TODO pole correction for pnabla_MYY
#             # ===========================
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "pnabla_MXX"),
#                 sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
#                     "pnabla_MXX"), "/", sir_utils.make_field_access_expr("vol")),
#             ),
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "pnabla_MYY"),
#                 sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
#                     "pnabla_MYY"), "/", sir_utils.make_field_access_expr("vol")),
#             ),
#         ]
#     )
