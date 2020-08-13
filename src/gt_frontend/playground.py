#from gtscript import stencil, Field, Vertex

#from . import gtscript

import inspect
import ast

from typing import Type
import gt_frontend.gtscript as gtscript
from gt_frontend.gtscript import Mesh, Field, Edge, Vertex
from gt_frontend.gtscript_to_gtir import GTScriptToGTIR, VarDeclExtractor
from gtc import common
from gt_frontend.frontend import GTScriptCompilationTask

dtype = common.DataType.FLOAT64

def edge_reduction(
    mesh: Mesh,
    edge_field: Field[Edge, dtype],
    vertex_field: Field[Vertex, dtype]
):
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        edge_field = sum(vertex_field[v] for v in vertices(e))
        #edge_field = 0.5 * sum(vertex_field[v] for v in vertices(e))
        #pass
        #edge_field[e] = 0.5*sum(vertex_field[v] for v in vertices(e))

GTScriptCompilationTask(edge_reduction).compile()
