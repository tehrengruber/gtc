# -*- coding: utf-8 -*-
from subprocess import PIPE, Popen
from types import MappingProxyType
from typing import ClassVar, Mapping

from mako import template as mako_tpl

from eve.codegen import TemplatedGenerator

from . import common
from .naive import LocationType


class NaiveCodeGenerator(TemplatedGenerator):
    LOCATION_TYPE_TO_STR: ClassVar[Mapping[LocationType, Mapping[str, str]]] = MappingProxyType(
        {
            LocationType.Node: MappingProxyType({"singular": "vertex", "plural": "vertices"}),
            LocationType.Edge: MappingProxyType({"singular": "edge", "plural": "edge"}),
            LocationType.Face: MappingProxyType({"singular": "cell", "plural": "cells"}),
        }
    )

    DATA_TYPE_TO_STR: ClassVar[Mapping[LocationType, str]] = MappingProxyType(
        {
            common.DataType.BOOLEAN: "bool",
            common.DataType.INT32: "int",
            common.DataType.UINT32: "unsigned_int",
            common.DataType.FLOAT32: "float",
            common.DataType.FLOAT64: "double",
        }
    )

    class Templates:
        UnstructuredField = mako_tpl.Template(
            """
<%
    loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]["singular"]
    data_type = _this_generator.DATA_TYPE_TO_STR[_this_node.data_type]
%>
  dawn::${ loc_type }_field_t<LibTag, ${ data_type }>& ${ name };"""
        )

        FieldAccessExpr = "{name}(deref(LibTag{{}}, {iter_var}), k)"

        AssignmentExpr = "{left} = {right}"

        ExprStmt = "\n{expr};"

        ForK = mako_tpl.Template(
            """
<%
    if _this_node.loop_order == _this_module.common.LoopOrder.FORWARD:
        k_init = '0'
        k_cond = 'k < k_size'
        k_step = '++k'
    else:
        k_init = 'k_size -1'
        k_cond = 'k >= 0'
        k_step = '--k'
%>
for (int k = ${k_init}; ${k_cond}; ${k_step}) {${ "".join(horizontal_loops) }\n}"""
        )

        HorizontalLoop = mako_tpl.Template(
            """
<%
    loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]['plural'].title()
%>
  for(auto const & t: get${ loc_type }(LibTag{}, mesh)) ${ ast }"""
        )

        BlockStmt = mako_tpl.Template("{${ ''.join(statements) }\n}")

        ReduceOverNeighbourExpr = mako_tpl.Template(
            """
<%
    right_loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.right_location_type]["singular"].title()
    loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]["singular"].title()
%>
            reduce${ right_loc_type }To${ loc_type }(LibTag{}, mesh, ${ outer_iter_var }, ${ init }, [&](auto& lhs, auto const& ${ iter_var }) {
return lhs ${ operation } = ${ right };
})"""
        )

        LiteralExpr = mako_tpl.Template(
            "(${ _this_generator.DATA_TYPE_TO_STR[_this_node.data_type] })${ value }"
        )

        Stencil = mako_tpl.Template(
            """
void ${name}() {
using dawn::deref;
${ "".join(k_loops) }
}
"""
        )

        Computation = mako_tpl.Template(
            """
<%
    stencil_calls = '\\n'.join("{name}();".format(name=s.name) for s in _this_node.stencils)
    ctor_field_params = ', '.join(
        'dawn::{loc_type}_field_t<LibTag, {data_type}>& {name}'.format(
            loc_type=_this_generator.LOCATION_TYPE_TO_STR[p.location_type]['singular'],
            name=p.name,
            data_type=_this_generator.DATA_TYPE_TO_STR[p.data_type],
        )
        for p in _this_node.params
    )
    ctor_field_initializers = ', '.join(
        '{name}({name})'.format(name=p.name) for p in _this_node.params
    )
%>
#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class generated {
private:
  dawn::mesh_t<LibTag> const& mesh;
  int const k_size;

${ ''.join(params) }
${ ''.join(stencils) }

public:
generated(dawn::mesh_t<LibTag> const& mesh, int k_size, ${ ctor_field_params }): mesh(mesh), k_size(k_size), ${ ctor_field_initializers } {}

void run() {
    ${ stencil_calls }
}
};
}
}

"""
        )

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        p = Popen(["clang-format"], stdout=PIPE, stdin=PIPE, encoding="utf8")
        formatted_code = p.communicate(input=generated_code)[0]
        return formatted_code

    def visit_HorizontalLoop(self, node, **kwargs) -> str:
        return self.generic_visit(node, iter_var="t", **kwargs)

    def visit_ReduceOverNeighbourExpr(self, node, *, iter_var, **kwargs) -> str:
        outer_iter_var = iter_var
        return self.generic_visit(node, outer_iter_var=outer_iter_var, iter_var="redIdx", **kwargs,)
