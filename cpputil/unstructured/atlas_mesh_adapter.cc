#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildDualMesh.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"
#include <atlas/grid/StructuredGrid.h>
#include <atlas/mesh.h>
#include <type_traits>

#include <gridtools/next/atlas_adapter.hpp>
#include <gridtools/next/mesh.hpp>

auto make_mesh() {
  atlas::StructuredGrid structuredGrid = atlas::Grid("O2");
  atlas::MeshGenerator::Parameters generatorParams;
  generatorParams.set("triangulate", true);
  generatorParams.set("angle", -1.0);

  atlas::StructuredMeshGenerator generator(generatorParams);

  return generator.generate(structuredGrid);
}

int main() {
  auto mesh = make_mesh();
  atlas::mesh::actions::build_edges(mesh);
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

  auto const &v2e =
      gridtools::next::mesh::connectivity<std::tuple<vertex, edge>>(mesh);

  auto n_vertices = gridtools::next::connectivity::primary_size(v2e);
  std::cout << n_vertices << std::endl;
  std::cout << gridtools::next::connectivity::max_neighbors(v2e) << std::endl;
  std::cout << gridtools::next::connectivity::skip_value(v2e) << std::endl;
  auto v2e_tbl = gridtools::next::connectivity::neighbor_table(v2e);

  static_assert(gridtools::is_sid<decltype(v2e_tbl)>{});

  auto strides = gridtools::sid::get_strides(v2e_tbl);

  std::cout << gridtools::at_key<neighbor>(strides) << std::endl;
  std::cout << gridtools::at_key<vertex>(strides) << std::endl;

  for (std::size_t v = 0; v < gridtools::next::connectivity::primary_size(v2e);
       ++v) {
    auto ptr = gridtools::sid::get_origin(v2e_tbl)();
    gridtools::sid::shift(ptr, gridtools::at_key<vertex>(strides), v);
    for (std::size_t i = 0;
         i < gridtools::next::connectivity::max_neighbors(v2e); ++i) {
      gridtools::sid::shift(ptr, gridtools::at_key<neighbor>(strides), 1);
    }
  }
}
