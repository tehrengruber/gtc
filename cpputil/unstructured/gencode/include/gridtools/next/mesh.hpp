#pragma once

#include <cstddef>
#include <gridtools/common/hymap.hpp>

namespace gridtools {
    namespace next {

        namespace connectivity {
            template <class Connectivity>
            auto neighbor_table(Connectivity const &connectivity) {
                return connectivity_neighbor_table(connectivity);
            };

            template <class Connectivity>
            auto max_neighbors(Connectivity const &connectivity) {
                return connectivity_max_neighbors(connectivity);
            }

            template <class Connectivity>
            std::size_t primary_size(Connectivity const &connectivity) {
                return connectivity_primary_size(connectivity);
            }

            template <class Connectivity>
            auto skip_value(Connectivity const &connectivity) {
                return connectivity_skip_value(connectivity);
            }
        } // namespace connectivity

        namespace mesh {
            template <typename Key, typename Mesh>
            decltype(auto) mesh_connectivity(const Mesh &mesh);

            template <class Key, class Mesh>
            decltype(auto) connectivity(Mesh const &mesh) {
                return mesh_connectivity<Key>(mesh);
            };
        } // namespace mesh

    } // namespace next
} // namespace gridtools
