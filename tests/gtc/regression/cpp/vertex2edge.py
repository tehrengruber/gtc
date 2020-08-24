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

from gtc.common import DataType
from gt_frontend.gtscript import Mesh, Field, Local, Cell, Edge, Vertex
from gt_frontend.frontend import GTScriptCompilationTask
from gtc.unstructured.ugpu_codegen import UgpuCodeGenerator
from gtc.unstructured.unaive_codegen import UnaiveCodeGenerator

dtype = DataType.FLOAT64

def sten(mesh : Mesh, field_in : Field[Vertex, dtype], field_out : Field[Edge, dtype]):
    with computation(FORWARD), location(Edge) as e:
        field_out = sum(field_in[v] for v in vertices(e))

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
    else: # 'ugpu':
        code_generator = UsidGpuCodeGenerator

    generated_code = GTScriptCompilationTask(sten).compile(debug=True, code_generator=code_generator)

    print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_vertex2edge_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)


if __name__ == "__main__":
    main()
