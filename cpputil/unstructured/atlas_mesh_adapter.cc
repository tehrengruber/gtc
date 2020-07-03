#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildDualMesh.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"
#include "gridtools/common/layout_map.hpp"
#include <atlas/grid/StructuredGrid.h>
#include <atlas/mesh.h>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>
#include <mesh/Connectivity.h>
#include <type_traits>

#include <gridtools/next/mesh.hpp>

namespace atlas {

template <class Key,
          std::enable_if_t<std::is_same_v<Key, std::tuple<vertex, edge>>, int> =
              0> // TODO protect
decltype(auto) mesh_connectivity(const Mesh &mesh) {
  return mesh.nodes().edge_connectivity();
}

} // namespace atlas

namespace gridtools::next::atlas_wrappers {

template <class LocationType, std::size_t MaxNeighbors>
struct regular_connectivity {
  static constexpr int NONE = -1;

  struct builder {
    auto operator()(std::size_t size) {
      return gridtools::storage::builder<gridtools::storage::cpu_ifirst>.type<int>().layout<0,1>().dimensions(size, std::integral_constant<std::size_t,
        MaxNeighbors>{});
    }
  };

  decltype(builder{}(std::size_t{})()) tbl_;

  regular_connectivity(atlas::mesh::IrregularConnectivity const &conn)
      : tbl_{builder{}(conn.rows())
                 .initializer([&conn](std::size_t row, std::size_t col) {
                   return col < conn.cols(row) ? conn.row(row)(col) : NONE;
                 })()} {}

  friend std::size_t
  connectivity_primary_size(regular_connectivity const &conn) {
    return conn.tbl_->lengths()[0];
  }

  friend std::integral_constant<std::size_t, MaxNeighbors>
  connectivity_max_neighbors(regular_connectivity const &conn) {
    return {};
  }

  friend int connectivity_skip_value(regular_connectivity const &conn) {
    return regular_connectivity::NONE;
  }
};

} // namespace gridtools::next::atlas_wrappers

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

  auto reg_con = gridtools::next::atlas_wrappers::regular_connectivity<edge, 7>{
      mesh.nodes().edge_connectivity()};

  auto n_vertices = gridtools::next::connectivity::primary_size(reg_con);
  //   auto v2e_tbl = gridtools::next::connectivity::neighbor_table(v2e);
  std::cout << n_vertices << std::endl;
  std::cout << gridtools::next::connectivity::max_neighbors(reg_con)
            << std::endl;
  std::cout << gridtools::next::connectivity::skip_value(reg_con) << std::endl;
}
