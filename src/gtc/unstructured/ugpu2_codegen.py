# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from types import MappingProxyType
from typing import ClassVar, Mapping

from devtools import debug  # noqa: F401
from mako import template as mako_tpl

from eve import NodeTranslator, codegen, NodeVisitor
from gtc import common
from gtc.unstructured.ugpu2 import (
    Computation,
    Connectivity,
    Kernel,
    KernelCall,
    SidCompositeNeighborTableEntry,
    SidCompositeEntry,
    Temporary,
)


# class SymbolTblHelper(NodeTranslator):
#     # TODO
#     # - temporary helper which resolves symbol refs with the symbol it's pointing to
#     # - the code generator relies on the possibility to look up a symbol ref outside of a visitor

#     def visit_SidCompositeNeighborTableEntry(self, node: SidCompositeNeighborTableEntry, **kwargs):
#         node.connectivity_deref_ = kwargs["symbol_tbl_conn"][node.connectivity]
#         return node

#     def visit_Kernel(self, node: Kernel, **kwargs):
#         symbol_tbl_conn = {c.name: c for c in node.connectivities}
#         symbol_tbl_sids = {s.name: s for s in node.sids}
#         return self.generic_visit(
#             node, symbol_tbl_conn=symbol_tbl_conn, symbol_tbl_sids=symbol_tbl_sids, **kwargs
#         )


class UgpuCodeGenerator(codegen.TemplatedGenerator):
    LOCATION_TYPE_TO_STR: ClassVar[
        Mapping[common.LocationType, Mapping[str, str]]
    ] = MappingProxyType(
        {
            common.LocationType.Vertex: "vertex",
            common.LocationType.Edge: "edge",
            common.LocationType.Cell: "cell",
        }
    )

    DATA_TYPE_TO_STR: ClassVar[Mapping[common.LocationType, str]] = MappingProxyType(
        {
            common.DataType.BOOLEAN: "bool",
            common.DataType.INT32: "int",
            common.DataType.UINT32: "unsigned_int",
            common.DataType.FLOAT32: "float",
            common.DataType.FLOAT64: "double",
        }
    )
    BUILTIN_LITERAL_TO_STR: ClassVar[Mapping[common.BuiltInLiteral, str]] = MappingProxyType(
        {
            common.BuiltInLiteral.MAX_VALUE: "std::numeric_limits<TODO>::max()",
            common.BuiltInLiteral.MIN_VALUE: "std::numeric_limits<TODO>::min()",
            common.BuiltInLiteral.ZERO: "0",
            common.BuiltInLiteral.ONE: "1",
        }
    )

    def location_type_from_dimensions(self, dimensions):
        location_type = [dim for dim in dimensions if isinstance(dim, common.LocationType)]
        if len(location_type) != 1:
            raise ValueError("Doesn't contain a LocationType!")
        return location_type[0]

    class Templates:
        # Node = mako_tpl.Template("${_this_node.__class__.__name__.upper()}")  # only for testing
        Connectivity = "auto {name} = gridtools::next::mesh::connectivity<{chain}>(mesh);"

        NeighborChain = mako_tpl.Template(
            """<%
                loc_strs = [_this_generator.LOCATION_TYPE_TO_STR[e] for e in _this_node.elements]
            %>
            % if len(loc_strs) == 1:
                ${ loc_strs[0] }
            % else:
                std::tuple<${ ','.join(loc_strs) }>
            % endif
            """
        )

        SidCompositeNeighborTableEntry = mako_tpl.Template(
            """<%
            connectivity_deref = symbol_tbl_conn[_this_node.connectivity]
            %>gridtools::next::connectivity::neighbor_table(${ connectivity_deref.name })"""
        )

        SidCompositeEntry = "{name}"

        SidComposite = mako_tpl.Template(
            # TODO the remaining change would be here: tag_name is now not available
            """
            auto ${ _this_node.field_name } = tu::make<gridtools::sid::composite::keys<${ ','.join([t.tag_name for t in _this_node.entries]) }>::values>(
            ${ ','.join(entries)});
            static_assert(gridtools::is_sid<decltype(${ _this_node.field_name })>{});
            """
        )

        KernelCall = mako_tpl.Template(
            """
            {
                ${ ''.join(connectivities) }

                ${ ''.join(sids) }

                auto [blocks, threads_per_block] = gridtools::next::cuda_util::cuda_setup(gridtools::next::connectivity::size(${ primary_connectivity.name }));
                ${ name }<<<blocks, threads_per_block>>>(${','.join(args)});
                GT_CUDA_CHECK(cudaDeviceSynchronize());
            }
            """
        )

        Kernel = mako_tpl.Template(
            """<%
                prim_conn = symbol_tbl_conn[_this_node.primary_connectivity]
                prim_sid = symbol_tbl_sids[_this_node.primary_sid]
            %>
            template<${ ','.join("class {}_t".format(p) for p in parameters)}>
            __global__ void ${ name }( ${','.join("{0}_t {0}".format(p) for p in parameters) }) {
                auto idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= gridtools::next::connectivity::size(${ prim_conn.name }))
                    return;
                auto ${ prim_sid.ptr_name } = ${ prim_sid.origin_name }();
                gridtools::sid::shift(${ prim_sid.ptr_name }, gridtools::device::at_key<
                    ${ _this_generator.LOCATION_TYPE_TO_STR[prim_sid.location.elements[-1]] }
                    >(${ prim_sid.strides_name }), idx);

                ${ "".join(ast) }
            }
            """
        )

        FieldAccess = mako_tpl.Template(
            """<%
                sid_deref = symbol_tbl_sids[_this_node.sid]
                sid_entry_deref = sid_deref.symbol_tbl[_this_node.name]
            %>*gridtools::device::at_key<${ sid_entry_deref.tag_name }>(${ sid_deref.ptr_name })"""
        )

        AssignStmt = "{left} = {right};"

        BinaryOp = "({left} {op} {right})"

        Computation = mako_tpl.Template(
            """#include <gridtools/next/cuda_util.hpp>
            #include <gridtools/next/mesh.hpp>
            #include <gridtools/next/tmp_gpu_storage.hpp>
            #include <gridtools/next/unstructured.hpp>
            #include <gridtools/common/cuda_util.hpp>
            #include <gridtools/sid/allocator.hpp>
            #include <gridtools/sid/composite.hpp>

            namespace ${ name }_impl_ {
                ${ ''.join(sid_tags) }

                ${ ''.join(kernels) }
            }

            template<class mesh_t, ${ ','.join('class ' + p.name + '_t' for p in _this_node.parameters) }>
            void ${ name }(mesh_t&& mesh, ${ ','.join(p.name + '_t&& ' + p.name for p in _this_node.parameters) }){
                namespace tu = gridtools::tuple_util;
                using namespace ${ name }_impl_;

                ${ cache_allocator if len(temporaries) > 0 else '' }
                ${ ''.join(temporaries) }

                ${ ''.join(ctrlflow_ast) }
            }
            """
        )

        Temporary = mako_tpl.Template(
            """<%
            %>auto zavg_tmp = gridtools::next::gpu::make_simple_tmp_storage<${ loctype }, ${ c_vtype }>(
                (int)gridtools::next::connectivity::size(gridtools::next::mesh::connectivity<${ loctype }>(mesh)), 1 /* TODO ksize */, cuda_alloc);"""
        )

        NeighborLoop = mako_tpl.Template(
            """<%
                outer_sid_deref = symbol_tbl_sids[_this_node.outer_sid]
                sid_deref = symbol_tbl_sids[_this_node.sid] if _this_node.sid else None
                conn_deref = symbol_tbl_conn[_this_node.connectivity]
                body_location = _this_generator.LOCATION_TYPE_TO_STR[sid_deref.location.elements[-1]] if sid_deref else None
            %>
            for (int neigh = 0; neigh < gridtools::next::connectivity::max_neighbors(${ conn_deref.name }); ++neigh) {
                auto absolute_neigh_index = *gridtools::device::at_key<${ conn_deref.neighbor_tbl_tag }>(${ outer_sid_deref.ptr_name});
                if (absolute_neigh_index != gridtools::next::connectivity::skip_value(${ conn_deref.name })) {
                    % if sid_deref:
                        auto ${ sid_deref.ptr_name } = ${ sid_deref.origin_name }();
                        gridtools::sid::shift(
                            ${ sid_deref.ptr_name }, gridtools::device::at_key<${ body_location }>(${ sid_deref.strides_name }), absolute_neigh_index);
                    % endif

                    // bodyparameters
                    ${ ''.join(body) }
                    // end body
                }
                gridtools::sid::shift(${ outer_sid_deref.ptr_name }, gridtools::device::at_key<neighbor>(${ outer_sid_deref.strides_name }), 1);
            }
            gridtools::sid::shift(${ outer_sid_deref.ptr_name }, gridtools::device::at_key<neighbor>(${ outer_sid_deref.strides_name }),
                -gridtools::next::connectivity::max_neighbors(${ conn_deref.name }));

            """
        )

        Literal = mako_tpl.Template(
            """<%
                literal= _this_node.value if isinstance(_this_node.value, str) else _this_generator.BUILTIN_LITERAL_TO_STR[_this_node.value]
            %>(${ _this_generator.DATA_TYPE_TO_STR[_this_node.vtype] })${ literal }"""
        )

        VarAccess = "{name}"

        VarDecl = mako_tpl.Template(
            "${ _this_generator.DATA_TYPE_TO_STR[_this_node.vtype] } ${ name } = ${ init };"
        )

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        symbol_tbl_resolved = root  # SymbolTblHelper().visit(root)
        generated_code = super().apply(symbol_tbl_resolved, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def visit_Temporary(self, node: Temporary, **kwargs):
        c_vtype = self.DATA_TYPE_TO_STR[node.vtype]
        loctype = self.LOCATION_TYPE_TO_STR[self.location_type_from_dimensions(node.dimensions)]
        return self.generic_visit(node, loctype=loctype, c_vtype=c_vtype, **kwargs)

    def visit_KernelCall(self, node: KernelCall, **kwargs):
        kernel: Kernel = kwargs["symbol_tbl_kernel"][node.name]
        connectivities = [self.generic_visit(conn, **kwargs) for conn in kernel.connectivities]
        primary_connectivity: Connectivity = kernel.symbol_tbl[kernel.primary_connectivity]
        sids = [self.generic_visit(s, **kwargs) for s in kernel.sids]

        # TODO I don't like that I render here and that I somehow have the same pattern for the parameters
        args = [c.name for c in kernel.connectivities]
        args += [
            "gridtools::sid::get_origin({0}), gridtools::sid::get_strides({0})".format(s.field_name)
            for s in kernel.sids
        ]
        # connectivity_args = [c.name for c in kernel.connectivities]
        return self.generic_visit(
            node,
            connectivities=connectivities,
            sids=sids,
            primary_connectivity=primary_connectivity,
            args=args,
            **kwargs,
        )

    def visit_Kernel(self, node: Kernel, **kwargs):
        symbol_tbl_conn = {c.name: c for c in node.connectivities}
        symbol_tbl_sids = {s.name: s for s in node.sids}

        parameters = [c.name for c in node.connectivities]
        for s in node.sids:
            parameters.append(s.origin_name)
            parameters.append(s.strides_name)

        return self.generic_visit(
            node,
            parameters=parameters,
            symbol_tbl_conn=symbol_tbl_conn,
            symbol_tbl_sids=symbol_tbl_sids,
            **kwargs,
        )

    def visit_Computation(self, node: Computation, **kwargs):
        class GetTagName(NodeVisitor):
            def __init__(self, **kwargs):
                super().__init__()
                self.result = set()
                self.symbol_tbl = dict()

            @classmethod
            def apply(cls, root, **kwargs):
                instance = cls()
                instance.visit(root)
                return instance.result

            def visit_SidCompositeNeighborTableEntry(
                self, node: SidCompositeNeighborTableEntry, **kwargs
            ):
                self.result.add(kwargs["symbol_tbl"][node.connectivity].neighbor_tbl_tag)

            def visit_SidCompositeEntry(self, node: SidCompositeEntry, **kwargs):
                self.result.add(node.tag_name)

            def visit_Kernel(self, node: Kernel, **kwargs):
                symbol_tbl = {conn.name: conn for conn in node.connectivities}
                self.visit(node.sids, symbol_tbl=symbol_tbl, **kwargs)

        print("----------------")
        print(GetTagName.apply(node))
        print("----------------")

        symbol_tbl_kernel = {k.name: k for k in node.kernels}

        sid_tags = GetTagName.apply(node)
        # sid_tags = set()
        # for k in node.kernels:
        #     for s in k.sids:
        #         for e in s.entries:
        #             sid_tags.add("struct " + e.tag_name + ";")
        cache_allocator = "auto cuda_alloc = gridtools::sid::device::make_cached_allocator(&gridtools::cuda_util::cuda_malloc<char[]>);"

        return self.generic_visit(
            node,
            computation_fields=node.parameters + node.temporaries,
            cache_allocator=cache_allocator,
            sid_tags=sid_tags,
            symbol_tbl_kernel=symbol_tbl_kernel,
            **kwargs,
        )
