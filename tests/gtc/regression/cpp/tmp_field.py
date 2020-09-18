# -*- coding: utf-8 -*-
#
# Copy stencil with temporary field

import os
import sys

from gt_frontend.frontend import GTScriptCompilationTask
from gt_frontend.gtscript import FORWARD, Cell, Field, Mesh, computation, location

from gtc.common import DataType
from gtc.unstructured.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


dtype = DataType.FLOAT64


def sten(mesh: Mesh, field_in: Field[Cell, dtype], field_out: Field[Cell, dtype]):
    with computation(FORWARD), interval(0, None), location(Cell):
        tmp = field_in
    with computation(FORWARD), interval(0, None), location(Cell):
        field_out = tmp


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator

    generated_code = GTScriptCompilationTask(sten).generate(
        debug=True, code_generator=code_generator
    )

    print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_tmp_field_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)


if __name__ == "__main__":
    main()
