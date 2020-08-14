import inspect
import ast

from typing import Type
import gt_frontend.gtscript as gtscript
from gt_frontend.gtscript import Mesh, Field, Edge, Vertex
from gt_frontend.gtscript_to_gtir import GTScriptToGTIR, VarDeclExtractor
from gtc import common
from gt_frontend.frontend import GTScriptCompilationTask

dtype = common.DataType.FLOAT64

#def edge_reduction(
#    mesh: Mesh,
#    edge_field: Field[Edge, dtype],
#    vertex_field: Field[Vertex, dtype]
#):
#    with computation(FORWARD), interval(0, None), location(Edge) as e:
#        edge_field = sum(vertex_field[v] for v in vertices(e))
#        #edge_field = 0.5 * sum(vertex_field[v] for v in vertices(e))
#        #pass
#        #edge_field[e] = 0.5*sum(vertex_field[v] for v in vertices(e))

def sparse_ex(
    mesh: Mesh,
    edge_field: Field[Edge, dtype],
    sparse_field: Field[Edge, Vertex, dtype]
):
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        edge_field = sum(sparse_field[e, v] for v in vertices(e))

def fvm_stencil(
    S: gtscript.Field[Edge, Vec[float, 2]],
    # zavgS: gtscript.Field[Edge, Vec[float, 2]],
    pp: gtscript.Field[Vertex, float],
    pnabla: gtscript.Field[Vertex, Vec[float, 2]],
    vol: gtscript.Field[Vertex, float],
    sign: gtscript.SparseField[Vertex, [Edge], float]
):
    with computation(FORWARD):
        with location(Edge) as e:
            zavg = 0.5 * sum(pp[v] for v in vertices(e))
            zavgS = S * zavg
        with location(Vertex) as v:
            pnabla = sum(zavgS[e] * sign[v, e] for e in edges(v))
            pnabla /= vol

GTScriptCompilationTask(sparse_ex).compile()
