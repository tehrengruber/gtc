# -*- coding: utf-8 -*-
# ignore "local variable '...' is assigned to but never used" for the entire file
# flake8: noqa: F841
from gt_frontend.frontend import GTScriptCompilationTask
from gt_frontend.gtscript import (
    FORWARD,
    Edge,
    Field,
    Mesh,
    Vertex,
    computation,
    interval,
    location,
    vertices,
)

from gtc import common


dtype = common.DataType.FLOAT64


def edge_reduction(mesh: Mesh, edge_field: Field[Edge, dtype], vertex_field: Field[Vertex, dtype]):
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        edge_field = sum(vertex_field[v] for v in vertices(e))
        # edge_field = 0.5 * sum(vertex_field[v] for v in vertices(e))
        # pass
        # edge_field[e] = 0.5*sum(vertex_field[v] for v in vertices(e))


def sparse_ex(
    mesh: Mesh, edge_field: Field[Edge, dtype], sparse_field: Field[Edge, Local[Vertex], dtype]
):
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        edge_field = sum(sparse_field[e, v] for v in vertices(e))


def test_nested(
    mesh: Mesh, f_1: Field[Edge, dtype], f_2: Field[Vertex, dtype], f_3: Field[Edge, dtype]
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            f_1 = 1
        with location(Vertex) as v:
            f_2 = 2
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        f_3 = 3


def fvm_nabla_stencil(
    mesh: Mesh,
    S_MXX: Field[Edge, dtype],
    S_MYY: Field[Edge, dtype],
    pp: Field[Vertex, dtype],
    pnabla_MXX: Field[Vertex, dtype],
    pnabla_MYY: Field[Vertex, dtype],
    vol: Field[Vertex, dtype],
    sign: Field[Vertex, Local[Edge], dtype],
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            zavg = 0.5 * sum(pp[v] for v in vertices(e))
            zavg = sum(pp[v] for v in vertices(e))
            zavgS_MXX = S_MXX * zavg
            zavgS_MYY = S_MYY * zavg
        with location(Vertex) as v:
            pnabla_MXX = sum(zavgS_MXX[e] * sign[v, e] for e in edges(v))
            pnabla_MYY = sum(zavgS_MYY[e] * sign[v, e] for e in edges(v))
            pnabla_MXX = pnabla_MXX / vol
            pnabla_MYY = pnabla_MYY / vol


GTScriptCompilationTask(fvm_nabla_stencil).generate()
