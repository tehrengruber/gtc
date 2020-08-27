#pragma once

#include <cstddef>
#include <utility>

#include "unstructured.hpp"
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/concept.hpp>

namespace gridtools {
    namespace next {

        namespace connectivity {
            // Doesn't need to be callable on device
            template <class Connectivity>
            GT_FUNCTION_HOST auto neighbor_table(Connectivity const &connectivity)
            /* -> decltype(connectivity_neighbor_table(connectivity)) */ // TODO this results in a CUDA error
            {
                return connectivity_neighbor_table(connectivity);
            };

            // template <class Connectivity>
            // using neighbor_table_type =
            //     decltype(::gridtools::next::connectivity::neighbor_table(std::declval<Connectivity const &>()));

            // template <class Connectivity>
            // using max_neighbors_type = gridtools::integral_constant<int,
            //     meta::second<meta::mp_find<
            //         hymap::to_meta_map<gridtools::sid::upper_bounds_type<neighbor_table_type<Connectivity>>>,
            //         neighbor>>>;

            // TODO remove(after refactoring)
            template <class Connectivity>
            GT_FUNCTION auto max_neighbors(Connectivity const &connectivity)
                -> decltype(connectivity_max_neighbors(connectivity)) {
                // return max_neighbors_type<Connectivity>::value;
                return connectivity_max_neighbors(connectivity);
            }

            template <class Connectivity>
            GT_FUNCTION std::size_t size(Connectivity const &connectivity) {
                return connectivity_size(connectivity);
            }

            template <class Connectivity>
            GT_FUNCTION auto skip_value(Connectivity const &connectivity)
                -> decltype(connectivity_skip_value(connectivity)) {
                return connectivity_skip_value(connectivity);
            }
        } // namespace connectivity

        namespace mesh {
            // models hymaps as Mesh
            // TODO(havogt): protection: only if values model the Connectivity concept and keys are meta::list<> of
            // locations
            // TODO probably remove?
            template <typename Key, typename Hymap> // TODO protect, only hymaps allowed
            decltype(auto) mesh_connectivity(Key const &, const Hymap &mesh) {
                return at_key<Key>(mesh);
            }

            template <class Key, class Mesh>
            auto connectivity(Mesh const &mesh) -> decltype(mesh_connectivity(meta::rename<meta::list, Key>(), mesh)) {
                return mesh_connectivity(meta::rename<meta::list, Key>(), mesh);
            }
        } // namespace mesh

    } // namespace next
} // namespace gridtools
