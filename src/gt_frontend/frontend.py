from typing import Type
import gt_frontend.gtscript as gtscript
from gt_frontend.gtscript import Mesh, Field, Edge, Vertex
from gt_frontend.gtscript_to_gtir import GTScriptToGTIR, VarDeclExtractor, CallCanonicalizer
from gtc import common
import inspect
import ast
from gtc.unstructured.gtir_to_nir import GtirToNir
from gtc.unstructured.nir_passes.merge_horizontal_loops import find_and_merge_horizontal_loops
from gtc.unstructured.nir_to_ugpu import NirToUgpu
from gtc.unstructured.ugpu_codegen import UgpuCodeGenerator

class GTScriptCompilationTask:
    def __init__(self, definition):
        self.symbol_table = {
            "dtype": common.DataType,
            "Vertex": common.LocationType,
            "Edge": common.LocationType,
            "Cell": common.LocationType
        }

        self.externals = {
            "dtype": common.DataType.FLOAT64,
            "Vertex": common.LocationType.Vertex,
            "Edge": common.LocationType.Edge,
            "Cell": common.LocationType.Cell
            #"Field": Field,
            #"Mesh": Mesh
        }

        self.definition = definition

    def compile(self):
        self._annotate_args()
        self.python_ast = ast.parse(inspect.getsource(self.definition)).body[0]
        self.gt4py_ast = gtscript.transform_py_ast_into_gt4py_ast(self.python_ast)

        VarDeclExtractor(self.symbol_table, self.externals).visit(self.gt4py_ast)
        CallCanonicalizer().visit(self.gt4py_ast)

        gtir = GTScriptToGTIR(self.symbol_table, self.externals).visit(self.gt4py_ast)

        nir_comp = GtirToNir().visit(gtir)
        nir_comp = find_and_merge_horizontal_loops(nir_comp)
        ugpu_comp = NirToUgpu().visit(nir_comp)
        # debug(ugpu_comp)

        generated_code = UgpuCodeGenerator.apply(ugpu_comp)
        print(generated_code)

        dump(gtir)

    def _annotate_args(self):
        """
        Populate symbol table by extracting the argument types from scope the function is embedded in
        """
        sig = inspect.signature(self.definition)
        for name, param in sig.parameters.items():
            self.symbol_table[name] = param.annotation
