#pragma once

#include <cstddef>

#include "gridtools/common/layout_map.hpp"
#include <atlas/mesh.h>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include <mesh/Connectivity.h>

#include "mesh.hpp"
#include "unstructured.hpp"
#include <gridtools/next/atlas_field_util.hpp>

#ifdef __CUDACC__ // TODO proper handling
#include <gridtools/storage/gpu.hpp>
using storage_trait = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_trait = gridtools::storage::cpu_ifirst;
#endif

namespace gridtools::next::atlas_wrappers {
    // not really a connectivity
    template <class LocationType>
    struct primary_connectivity {
        std::size_t size_;

        GT_FUNCTION friend std::size_t connectivity_size(primary_connectivity const &conn) { return conn.size_; }
    };

    template <class LocationType, std::size_t MaxNeighbors>
    struct regular_connectivity {
        struct builder {
            auto operator()(std::size_t size) {
                return gridtools::storage::builder<storage_trait>.template type<int>().template layout<0, 1>().template dimensions(
                    size, std::integral_constant<std::size_t, MaxNeighbors>{});
            }
        };

        decltype(builder{}(std::size_t{})()) tbl_;
        const atlas::idx_t missing_value_; // TODO Not sure if we can leave the type open
        const gridtools::uint_t size_;

        regular_connectivity(atlas::mesh::IrregularConnectivity const &conn)
            : tbl_{builder{}(conn.rows()).initializer([&conn](std::size_t row, std::size_t col) {
                  return col < conn.cols(row) ? conn.row(row)(col) : conn.missing_value();
              })()},
              missing_value_{conn.missing_value()}, size_{tbl_->lengths()[0]} {}

        regular_connectivity(atlas::mesh::MultiBlockConnectivity const &conn)
            : tbl_{builder{}(conn.rows()).initializer([&conn](std::size_t row, std::size_t col) {
                  return col < conn.cols(row) ? conn.row(row)(col) : conn.missing_value();
              })()},
              missing_value_{conn.missing_value()}, size_{tbl_->lengths()[0]} {}

        //   template <class Filter>
        //   auto initialize_filtered(atlas::mesh::MultiBlockConnectivity const &conn,
        //                            Filter &&filter) {
        //     static_assert(
        //         false,
        //         "This idea is broken, because I don't know which edge I am on...");
        //     std::vector<atlas::idx_t> tmp;
        //     for (atlas::idx_t i = 0; i < conn.rows(); ++i) {
        //       if (filter(i))
        //         tmp.push_back(i);
        //     }
        //     return builder{}(tmp.size())
        //         .initializer([&conn, &tmp](std::size_t row, std::size_t col) {
        //           return col < conn.cols(tmp[row]) ? conn.row(tmp[row])(col)
        //                                            : conn.missing_value();
        //         })();
        //   }

        //   template <class Filter>
        //   regular_connectivity(atlas::mesh::MultiBlockConnectivity const &conn,
        //                        Filter &&filter)
        //       : tbl_{initialize_filtered(conn, std::forward<Filter>(filter))},
        //         missing_value_{conn.missing_value()} {}

        GT_FUNCTION friend std::size_t connectivity_size(regular_connectivity const &conn) { return conn.size_; }

        GT_FUNCTION friend std::integral_constant<std::size_t, MaxNeighbors> connectivity_max_neighbors(
            regular_connectivity const &conn) {
            return {};
        }

        GT_FUNCTION friend int connectivity_skip_value(regular_connectivity const &conn) { return conn.missing_value_; }

        friend auto connectivity_neighbor_table(regular_connectivity const &conn) {
            return gridtools::sid::rename_numbered_dimensions<LocationType, neighbor>(conn.tbl_);
        }
    };

    // template <class LocationType, std::size_t MaxNeighbors>
    // struct sparse_connectivity {
    //   struct builder {
    //     auto operator()(std::size_t size) {
    //       return
    //       gridtools::storage::builder<gridtools::storage::cpu_ifirst>.type<int>().layout<0,1>().dimensions(size,
    //       std::integral_constant<std::size_t,
    //         MaxNeighbors>{});
    //     }
    //   };

    //   decltype(builder{}(std::size_t{})()) elements_;
    //   decltype(builder{}(std::size_t{})()) tbl_;
    //   const atlas::idx_t
    //       missing_value_; // TODO Not sure if we can leave the type open

    //   template <class Filter>
    //   auto initialize_filtered(atlas::mesh::MultiBlockConnectivity const &conn,
    //                            Filter &&filter) {
    //     std::vector<atlas::idx_t> tmp;
    //     for (atlas::idx_t i = 0; i < conn.rows(); ++i) {
    //       if (filter(i))
    //         tmp.push_back(i);
    //     }
    //     return builder{}(tmp.size())
    //         .initializer([&conn, &tmp](std::size_t row, std::size_t col) {
    //           return col < conn.cols(tmp[row]) ? conn.row(tmp[row])(col)
    //                                            : conn.missing_value();
    //         })();
    //   }

    //   template <class Filter>
    //   sparse_connectivity(atlas::mesh::MultiBlockConnectivity const &conn,
    //                       Filter &&filter)
    //       : elements_{}, tbl_{initialize_filtered(conn,
    //       std::forward<Filter>(filter))},
    //         missing_value_{conn.missing_value()} {}

    //   friend std::size_t
    //   connectivity_size(sparse_connectivity const &conn) {
    //     return conn.tbl_->lengths()[0];
    //   }

    //   friend std::integral_constant<std::size_t, MaxNeighbors>
    //   connectivity_max_neighbors(sparse_connectivity const &conn) {
    //     return {};
    //   }

    //   friend int connectivity_skip_value(sparse_connectivity const &conn) {
    //     return conn.missing_value_;
    //   }

    //   friend auto connectivity_neighbor_table(sparse_connectivity const &conn) {

    //     // return gridtools::sid::rename_all_dimensions<
    //     //     gridtools::hymap::keys<LocationType, neighbor>>(conn.tbl_);
    //     return gridtools::sid::rename_dimension<
    //         gridtools::integral_constant<int, 1>, neighbor>(
    //         gridtools::sid::rename_dimension<gridtools::integral_constant<int,
    //         0>,
    //                                          LocationType>(conn.tbl_));
    //   }
    // };

} // namespace gridtools::next::atlas_wrappers

namespace atlas {

    template <class Key, std::enable_if_t<std::is_same_v<Key, std::tuple<vertex, edge>>, int> = 0> // TODO protect
    decltype(auto) mesh_connectivity(const Mesh &mesh) {
        return gridtools::next::atlas_wrappers::regular_connectivity<vertex, 7
            // TODO this number must passed by the user (probably wrap atlas mesh)
            >{mesh.nodes().edge_connectivity()};
    }

    template <class Key, std::enable_if_t<std::is_same_v<Key, std::tuple<edge, vertex>>, int> = 0> // TODO protect
    decltype(auto) mesh_connectivity(const Mesh &mesh) {
        return gridtools::next::atlas_wrappers::regular_connectivity<edge, 2>{mesh.edges().node_connectivity()};
    }

    template <class Key, std::enable_if_t<std::is_same_v<Key, edge>, int> = 0> // TODO protect
    decltype(auto) mesh_connectivity(const Mesh &mesh) {
        return gridtools::next::atlas_wrappers::primary_connectivity<edge>{mesh.edges().size()};
    }
    template <class Key, std::enable_if_t<std::is_same_v<Key, vertex>, int> = 0> // TODO protect
    decltype(auto) mesh_connectivity(const Mesh &mesh) {
        return gridtools::next::atlas_wrappers::primary_connectivity<vertex>{mesh.nodes().size()};
    }

    // struct pole_edge;

    // template <class Key,
    //           std::enable_if_t<std::is_same_v<Key, std::tuple<pole_edge,
    //           vertex>>,
    //                            int> = 0> // TODO protect
    // decltype(auto) mesh_connectivity(const Mesh &mesh) {
    //   const auto edge_flags = array::make_view<int, 1>(mesh.edges().flags());
    //   return gridtools::next::atlas_wrappers::regular_connectivity<pole_edge, 2>{
    //       mesh.edges().node_connectivity(), [&edge_flags](atlas::idx_t edge) {
    //         return atlas::mesh::Nodes::Topology::check(
    //             edge_flags(edge), atlas::mesh::Nodes::Topology::POLE);
    //       }};
    // }

} // namespace atlas