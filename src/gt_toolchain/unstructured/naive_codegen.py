# -*- coding: utf-8 -*-
from subprocess import PIPE, Popen

from eve.codegen import TemplatedGenerator

from . import common
from .naive import LocationType


def location_type_to_str(location_type):
    if location_type == LocationType.Node:
        return "Vertices"
    elif location_type == LocationType.Edge:
        return "Edges"
    elif location_type == LocationType.Face:
        return "Cells"
    else:
        raise "unreachable"


def location_type_to_str_singular(location_type):
    if location_type == LocationType.Node:
        return "Vertex"
    elif location_type == LocationType.Edge:
        return "Edge"
    elif location_type == LocationType.Face:
        return "Cell"
    else:
        raise "unreachable"


def location_type_to_str_singular_lower(location_type):
    if location_type == LocationType.Node:
        return "vertex"
    elif location_type == LocationType.Edge:
        return "edge"
    elif location_type == LocationType.Face:
        return "cell"
    else:
        raise "unreachable"


def data_type_to_c(data_type):
    if data_type == common.DataType.BOOLEAN:
        return "bool"
    elif data_type == common.DataType.INT32:
        return "int"
    elif data_type == common.DataType.UINT32:
        return "unsigned int"
    elif data_type == common.DataType.FLOAT32:
        return "float"
    elif data_type == common.DataType.FLOAT64:
        return "double"
    else:
        raise "unreachable"


class NaiveCodeGenerator(TemplatedGenerator):
    def __init__(self, node_templates, dump_func):
        super().__init__(node_templates, dump_func)
        self.iter_var_stack = ["NONE"]

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        p = Popen(["clang-format"], stdout=PIPE, stdin=PIPE, encoding="utf8")
        formatted_code = p.communicate(input=generated_code)[0]
        return formatted_code

    NODE_TEMPLATES = dict(
        UnstructuredField="\n  dawn::{loctype}_field_t<LibTag, {data_type}>& {name};",
        FieldAccessExpr="{name}(deref(LibTag{{}}, {iter_var}), k)",
        AssignmentExpr="{left} = {right}",
        ExprStmt="\n{expr};",
        ForK="{k_loop} {{{horizontal_loops}\n}}",
        HorizontalLoop="\nfor(auto const & t: get{stringified_location_type}(LibTag{{}}, mesh)) {ast}",
        BlockStmt="{{{statements}\n}}",
        ReduceOverNeighbourExpr="""reduce{right_loc_type}To{loc_type}(LibTag{{}}, mesh, {outer_iter_var}, {init}, [&](auto& lhs, auto const& {inner_iter_var}) {{
return lhs {operation}= {right};
}})""",
        LiteralExpr="({c_data_type}){value}",
        Stencil="""
void {name}() {{
using dawn::deref;
{k_loops}
}}
""",
        Computation="""#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {{
namespace cxxnaiveico {{
template <typename LibTag>
class generated {{
private:
  dawn::mesh_t<LibTag> const& mesh;
  int const k_size;

{api_field_declarations}
{stencil_definitions}

public:
generated(dawn::mesh_t<LibTag> const& mesh, int k_size, {ctor_field_params}): mesh(mesh), k_size(k_size), {ctor_field_initializers} {{}}

void run() {{
    {stencil_calls}
}}
}};
}}
}}

""",
    )

    def visit_UnstructuredField(self, node, **kwargs) -> str:
        return self.render_node(
            node,
            dict(
                loctype=location_type_to_str_singular_lower(node.location_type),
                data_type=data_type_to_c(node.data_type),
            ),
        )

    def visit_FieldAccessExpr(self, node, **kwargs) -> str:
        return self.render_node(node, dict(iter_var=self.iter_var_stack[-1]))

    def visit_ForK(self, node, **kwargs) -> str:
        if node.loop_order == common.LoopOrder.FORWARD:
            k_loop = "\nfor(int k = 0; k < k_size; ++k)"
        else:
            k_loop = "\nfor(int k = k_size-1; k >= 0; --k)"

        horizontal_loops = "".join(self.visit(h) for h in node.horizontal_loops)
        return self.render_node(node, dict(horizontal_loops=horizontal_loops, k_loop=k_loop))

    def visit_HorizontalLoop(self, node, **kwargs) -> str:
        stringified_location_type = location_type_to_str(node.location_type)
        self.iter_var_stack.append("t")
        ast = self.visit(node.ast)
        return self.render_node(
            node, dict(stringified_location_type=stringified_location_type, ast=ast)
        )

    def visit_BlockStmt(self, node, **kwargs) -> str:
        statements = "".join(self.visit(s) for s in node.statements)
        return self.render_node(node, dict(statements=statements))

    def visit_ReduceOverNeighbourExpr(self, node, **kwargs) -> str:
        right_loc_type = location_type_to_str_singular(node.right_location_type)
        loc_type = location_type_to_str_singular(node.location_type)
        init = self.visit(node.init)
        operation = self.visit(node.operation)
        outer_iter_var = self.iter_var_stack[-1]
        self.iter_var_stack.append("redIdx")
        inner_iter_var = self.iter_var_stack[-1]
        right = self.visit(node.right)
        self.iter_var_stack.pop()

        return self.render_node(
            node,
            dict(
                right_loc_type=right_loc_type,
                loc_type=loc_type,
                init=init,
                operation=operation,
                right=right,
                outer_iter_var=outer_iter_var,
                inner_iter_var=inner_iter_var,
            ),
        )

    def visit_LiteralExpr(self, node, **kwargs) -> str:
        return self.render_node(
            node, dict(c_data_type=data_type_to_c(node.data_type), value=self.visit(node.value))
        )

    def visit_Stencil(self, node, **kwargs) -> str:
        k_loops = "".join(self.visit(l) for l in node.k_loops)
        return self.render_node(node, dict(k_loops=k_loops))

    def visit_Computation(self, node, **kwargs) -> str:
        api_field_declarations = "".join(self.visit(p) for p in node.params)
        stencil_definitions = "".join(self.visit(s) for s in node.stencils)
        ctor_field_params = ", ".join(
            "dawn::{loc_type}_field_t<LibTag, {data_type}>& {name}".format(
                loc_type=location_type_to_str_singular_lower(p.location_type),
                name=p.name,
                data_type=data_type_to_c(p.data_type),
            )
            for p in node.params
        )
        ctor_field_initializers = ", ".join(
            "{name}({name})".format(name=p.name) for p in node.params
        )
        stencil_calls = "\n".join("{name}();".format(name=s.name) for s in node.stencils)
        return self.render_node(
            node,
            dict(
                api_field_declarations=api_field_declarations,
                stencil_definitions=stencil_definitions,
                ctor_field_params=ctor_field_params,
                ctor_field_initializers=ctor_field_initializers,
                stencil_calls=stencil_calls,
            ),
        )
