#pragma once

#include <cstddef>

#include "gridtools/common/layout_map.hpp"
#include <atlas/mesh.h>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/rename_dimension.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include <gridtools/next/mesh.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>

namespace gridtools::next::atlas_wrappers {

template <class LocationType, std::size_t MaxNeighbors>
struct regular_connectivity {
  struct builder {
    auto operator()(std::size_t size) {
      return gridtools::storage::builder<gridtools::storage::cpu_ifirst>.type<int>().layout<0,1>().dimensions(size, std::integral_constant<std::size_t,
        MaxNeighbors>{});
    }
  };

  decltype(builder{}(std::size_t{})()) tbl_;
  const atlas::idx_t missing_value_; // Not sure if we can leave the type open

  regular_connectivity(atlas::mesh::IrregularConnectivity const &conn)
      : tbl_{builder{}(conn.rows())
                 .initializer([&conn](std::size_t row, std::size_t col) {
                   return col < conn.cols(row) ? conn.row(row)(col)
                                               : conn.missing_value();
                 })()},
        missing_value_{conn.missing_value()} {}

  friend std::size_t
  connectivity_primary_size(regular_connectivity const &conn) {
    return conn.tbl_->lengths()[0];
  }

  friend std::integral_constant<std::size_t, MaxNeighbors>
  connectivity_max_neighbors(regular_connectivity const &conn) {
    return {};
  }

  friend int connectivity_skip_value(regular_connectivity const &conn) {
    return conn.missing_value_;
  }

  friend auto connectivity_neighbor_table(regular_connectivity const &conn) {

    return gridtools::sid::rename_dimension<
        gridtools::integral_constant<int, 1>, neighbor>(
        gridtools::sid::rename_dimension<gridtools::integral_constant<int, 0>,
                                         LocationType>(conn.tbl_));
  }
};

} // namespace gridtools::next::atlas_wrappers

namespace atlas {

template <class Key,
          std::enable_if_t<std::is_same_v<Key, std::tuple<vertex, edge>>, int> =
              0> // TODO protect
decltype(auto) mesh_connectivity(const Mesh &mesh) {
  return gridtools::next::atlas_wrappers::regular_connectivity<
      vertex, 7
      // TODO this number must passed by the user (probably wrap atlas mesh)
      >{mesh.nodes().edge_connectivity()};
}

} // namespace atlas
