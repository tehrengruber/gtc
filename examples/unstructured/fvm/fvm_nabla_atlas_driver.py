# -*- coding: utf-8 -*-
import numpy as np
from atlas4py import *


grid = StructuredGrid("O32")
# grid = StructuredGrid(x_spacing=LinearSpacing(-1, 1, 20), y_spacing=LinearSpacing(-1, 1, 21))
# print(grid)

mesh = StructuredMeshGenerator().generate(grid)  # Generator parameters

fs_edges = functionspace.EdgeColumns(mesh, halo=1)  # TODO nb_levels
fs_nodes = functionspace.NodeColumns(mesh, halo=1)
print(fs_nodes)

build_edges(mesh)
build_node_to_edge_connectivity(mesh)
build_median_dual_mesh(mesh)

m_S_MXX = fs_edges.create_field(
    name="m_S_MXX", levels=1, dtype=np.float64
)  # TODO why is levels here and not on the function space?
m_S_MYY = fs_edges.create_field(name="m_S_MYY", levels=1, dtype=np.float64)
vol = fs_nodes.create_field(name="vol", levels=1, dtype=np.float64)
edges_per_node = 7
m_sign = fs_nodes.create_field(name="m_sign", levels=1, dtype=np.float64, variables=edges_per_node)
#           atlas::MeshGenerator::Parameters generatorParams;
#           generatorParams.set("triangulate", true);
#           generatorParams.set("angle", -1.0);
#           atlas::StructuredMeshGenerator generator(generatorParams);
#         fs_edges_(mesh_, atlas::option::levels(nb_levels) | atlas::option::halo(1)),
#         fs_nodes_(mesh_, atlas::option::levels(nb_levels) | atlas::option::halo(1)), //

# ============ initialize_S()
# all fields supported by dawn are 2 (or 3 with sparse) dimensional:
# (unstructured, lev, sparse) S has dimensions (unstructured, [MMX/MMY])
# const auto S =
#     atlas::array::make_view<double, 2>(mesh_.edges().field("dual_normals"));
S = np.array(mesh.edges.field("dual_normals"), copy=False)
S_MXX = np.array(m_S_MXX, copy=False)
print(S[100, 1])
print(S_MXX[0, 0])
# auto S_MXX = atlas::array::make_view<double, 2>(m_S_MXX);
# auto S_MYY = atlas::array::make_view<double, 2>(m_S_MYY);
klevel = 0
MXX = 0
MYY = 1
print(mesh.edges.size)
print(S_MXX)
for i in range(0, mesh.edges.size - 1):
    S_MXX[i, klevel] = S[i, MXX]

# assert(nb_levels_ == 1);
# int klevel = 0;
# for (int i = 0, size = mesh_.edges().size(); i < size; ++i) {
#   S_MXX(i, klevel) = S(i, MXX) * radius * deg2rad;
#   S_MYY(i, klevel) = S(i, MYY) * radius * deg2rad;
# }
# =========end initialize_S()
