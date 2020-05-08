# -*- coding: utf-8 -*-
import math
import os
import subprocess

import cppimport.import_hook
import numpy as np
from atlas4py import (
    Config,
    StructuredGrid,
    StructuredMeshGenerator,
    Topology,
    build_edges,
    build_median_dual_mesh,
    build_node_to_edge_connectivity,
    functionspace,
)


output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_fvm_nabla.hpp"
with open(output_file, "w+") as output:
    subprocess.call(
        ["python", "fvm_nabla.py"], stdout=output, cwd=os.path.dirname(os.path.realpath(__file__)),
    )

import fvm_nabla_wrapper  # noqa: E402 isort:skip

cppimport.force_rebuild(True)

grid = StructuredGrid("O32")
config = Config()
config["triangulate"] = True
config["angle"] = 20.0
# config["angle"] = -1.0
# TODO I am a bit confused: 20.0 and -1.0 produce different results,
# sometimes one sometimes the other seems to be the correct value to reproduce Christian's results
mesh = StructuredMeshGenerator(config).generate(grid)  # Generator parameters

fs_edges = functionspace.EdgeColumns(mesh, halo=1)  # TODO nb_levels
fs_nodes = functionspace.NodeColumns(mesh, halo=1)

build_edges(mesh)
build_node_to_edge_connectivity(mesh)
build_median_dual_mesh(mesh)

m_S_MXX = fs_edges.create_field(
    name="m_S_MXX", levels=1, dtype=np.float64
)  # TODO why is levels here and not on the function space?
m_S_MYY = fs_edges.create_field(name="m_S_MYY", levels=1, dtype=np.float64)
m_vol = fs_nodes.create_field(name="vol", levels=1, dtype=np.float64)
edges_per_node = 7
m_sign = fs_nodes.create_field(name="m_sign", levels=1, dtype=np.float64, variables=edges_per_node)

# ============ initialize_S()
# all fields supported by dawn are 2 (or 3 with sparse) dimensional:
# (unstructured, lev, sparse) S has dimensions (unstructured, [MMX/MMY])
S = np.array(mesh.edges.field("dual_normals"), copy=False)
S_MXX = np.array(m_S_MXX, copy=False)
S_MYY = np.array(m_S_MYY, copy=False)

klevel = 0
MXX = 0
MYY = 1

rpi = 2.0 * math.asin(1.0)
radius = 6371.22e03
deg2rad = 2.0 * rpi / 360.0

for i in range(0, fs_edges.size - 1):
    S_MXX[i, klevel] = S[i, MXX] * radius * deg2rad
    S_MYY[i, klevel] = S[i, MYY] * radius * deg2rad


def min_max_avg(field):
    view = np.array(field, copy=False)
    print(
        "{}:  min={}, max={}, avg={}".format(
            field.name, min(view), max(view), sum(view) / len(view)
        )
    )


#  SXX(min/max):  -103437.60479272791    340115.33913622628
#  SYY(min/max):  -2001577.7946404363    2001577.7946404363
min_max_avg(m_S_MXX)
min_max_avg(m_S_MYY)
# =========end initialize_S()
# ============ initialize_sign()
node2edge_sign = np.array(m_sign, copy=False)
edge_flags = np.array(mesh.edges.flags())


def is_pole_edge(e):
    return Topology.check(edge_flags[e], Topology.POLE)


for jnode in range(0, fs_nodes.size - 1):
    node_edge_con = mesh.nodes.edge_connectivity
    edge_node_con = mesh.edges.node_connectivity
    for jedge in range(0, node_edge_con.cols(jnode) - 1):
        iedge = node_edge_con[jnode, jedge]
        ip1 = edge_node_con[iedge, 0]
        if jnode == ip1:
            node2edge_sign[jnode, 0, jedge] = 1.0
        else:
            node2edge_sign[jnode, 0, jedge] = -1.0
            if is_pole_edge(iedge):
                node2edge_sign[jnode, 0, jedge] = 1.0
# =========end initialize_sign()
# ============ initialize_vol()
vol_atlas = np.array(mesh.nodes.field("dual_volumes"), copy=False)
# dual_volumes min=2.32551, max=68.8916
min_max_avg(mesh.nodes.field("dual_volumes"))
vol = np.array(m_vol, copy=False)
for i in range(0, vol_atlas.size):
    vol[i, klevel] = vol_atlas[i] * pow(deg2rad, 2) * pow(radius, 2)
# VOL(min/max):  57510668192.214096    851856184496.32886
min_max_avg(m_vol)
# =========vol initialize_vol()

# ============ fillInputData()
zh0 = 2000.0
zrad = 3.0 * rpi / 4.0 * radius
zeta = rpi / 16.0 * radius
zlatc = 0.0
zlonc = 3.0 * rpi / 2.0


# =========end fillInputData()
m_rlonlatcr = fs_nodes.create_field(
    name="m_rlonlatcr", levels=1, dtype=np.float64, variables=edges_per_node
)
rlonlatcr = np.array(m_rlonlatcr, copy=False)

m_rcoords = fs_nodes.create_field(
    name="m_rcoords", levels=1, dtype=np.float64, variables=edges_per_node
)
rcoords = np.array(m_rcoords, copy=False)

m_rcosa = fs_nodes.create_field(name="m_rcosa", levels=1, dtype=np.float64)
rcosa = np.array(m_rcosa, copy=False)

m_rsina = fs_nodes.create_field(name="m_rsina", levels=1, dtype=np.float64)
rsina = np.array(m_rsina, copy=False)

m_pp = fs_nodes.create_field(name="m_pp", levels=1, dtype=np.float64)
rzs = np.array(m_pp, copy=False)

rcoords_deg = np.array(mesh.nodes.field("lonlat"))

for jnode in range(0, fs_nodes.size - 1):
    for i in range(0, 1):
        rcoords[jnode, klevel, i] = rcoords_deg[jnode, i] * deg2rad
        rlonlatcr[jnode, klevel, i] = rcoords[jnode, klevel, i]  # This is not my pattern!
    rcosa[jnode, klevel] = math.cos(rlonlatcr[jnode, klevel, MYY])
    rsina[jnode, klevel] = math.sin(rlonlatcr[jnode, klevel, MYY])
for jnode in range(0, fs_nodes.size - 1):
    zlon = rlonlatcr[jnode, klevel, MXX]
    zdist = math.sin(zlatc) * rsina[jnode, klevel] + math.cos(zlatc) * rcosa[
        jnode, klevel
    ] * math.cos(zlon - zlonc)
    zdist = radius * math.acos(zdist)
    rzs[jnode, klevel] = 0.0
    if zdist < zrad:
        rzs[jnode, klevel] = rzs[jnode, klevel] + 0.5 * zh0 * (
            1.0 + math.cos(rpi * zdist / zrad)
        ) * math.pow(math.cos(rpi * zdist / zeta), 2)

min_max_avg(m_pp)  # TODO max is too high
min_max_avg(m_pp)

m_zavgS_MXX = fs_edges.create_field(name="m_zavgS_MXX", levels=1, dtype=np.float64)
m_zavgS_MYY = fs_edges.create_field(name="m_zavgS_MYY", levels=1, dtype=np.float64)
m_pnabla_MXX = fs_nodes.create_field(name="m_pnabla_MXX", levels=1, dtype=np.float64)
m_pnabla_MYY = fs_nodes.create_field(name="m_pnabla_MYY", levels=1, dtype=np.float64)

fvm_nabla_wrapper.run_computation(
    mesh,
    1,
    m_S_MXX,
    m_S_MYY,
    m_zavgS_MXX,
    m_zavgS_MYY,
    m_pp,
    m_pnabla_MXX,
    m_pnabla_MYY,
    m_vol,
    m_sign,
)

# zavgSXX(min/max):  -199755464.25741270    388241977.58389181
# zavgSYY(min/max):  -1000788897.3202186    1000788897.3202186
min_max_avg(m_zavgS_MXX)
min_max_avg(m_zavgS_MYY)
#  nablaXX(min/max):  -3.5455427772566003E-003  3.5455427772565435E-003
#  nablaYY(min/max):  -3.3540113705465301E-003  3.3540113705465301E-003
min_max_avg(m_pnabla_MXX)
min_max_avg(m_pnabla_MYY)
