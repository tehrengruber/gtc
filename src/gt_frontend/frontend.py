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

import ast
import inspect
from typing import Type

import devtools
import gt_frontend.gtscript as gtscript
from gt_frontend.gtscript import Edge, Field, Mesh, Vertex
from gt_frontend.gtscript_to_gtir import (
    GTScriptToGTIR,
    NodeCanonicalizer,
    SymbolResolutionValidation,
    SymbolTable,
    TemporaryFieldDeclExtractor,
    VarDeclExtractor,
)
from gt_frontend.py_to_gtscript import PyToGTScript

from gtc import common
from gtc.unstructured.gtir_to_nir import GtirToNir
from gtc.unstructured.nir_passes.merge_horizontal_loops import find_and_merge_horizontal_loops
from gtc.unstructured.nir_to_ugpu import NirToUgpu
from gtc.unstructured.ugpu_codegen import UgpuCodeGenerator


class GTScriptCompilationTask:
    def __init__(self, definition):
        self.symbol_table = SymbolTable(
            types={
                "dtype": common.DataType,
                "Vertex": common.LocationType,
                "Edge": common.LocationType,
                "Cell": common.LocationType,
            },
            constants={
                "dtype": common.DataType.FLOAT64,
                "Vertex": common.LocationType.Vertex,
                "Edge": common.LocationType.Edge,
                "Cell": common.LocationType.Cell,
                # "Field": Field,
                # "Mesh": Mesh
            },
        )

        self.definition = definition

    def _annotate_args(self):
        """
        Populate symbol table by extracting the argument types from scope the function is embedded in
        """
        sig = inspect.signature(self.definition)
        for name, param in sig.parameters.items():
            self.symbol_table[name] = param.annotation

    def compile(self, *, debug=False, code_generator=UgpuCodeGenerator):
        self._annotate_args()
        self.python_ast = ast.parse(inspect.getsource(self.definition)).body[0]
        self.gt4py_ast = PyToGTScript().transform(self.python_ast)

        # Canonicalization
        NodeCanonicalizer.apply(self.gt4py_ast)

        # Populate symbol table
        VarDeclExtractor.apply(self.symbol_table, self.gt4py_ast)
        TemporaryFieldDeclExtractor.apply(self.symbol_table, self.gt4py_ast)
        SymbolResolutionValidation.apply(self.symbol_table, self.gt4py_ast)

        # Transform into GTIR
        gtir = GTScriptToGTIR.apply(self.symbol_table, self.gt4py_ast)

        # Code generation
        nir_comp = GtirToNir().visit(gtir)
        nir_comp = find_and_merge_horizontal_loops(nir_comp)
        usid_comp = NirToUsid().visit(nir_comp)

        if debug:
            devtools.debug(nir_comp)
            devtools.debug(usid_comp)

        generated_code = code_generator.apply(usid_comp)

        return generated_code
