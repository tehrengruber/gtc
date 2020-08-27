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
            struct not_provided;
            not_provided mesh_connectivity(...);

            // Models gridtools::hymaps as mesh
            template <class Key,
                class Mesh,
                class K = meta::rename<meta::list, Key>,
                class Res = decltype(mesh_connectivity(K(), std::declval<Mesh const &>())),
                std::enable_if_t< //
                    std::is_same_v<Res, not_provided>
                    /* && is_connectivity_v<decltype(at_key<K>(std::declval<Mesh const&>()))> */,
                    int> = 0>
            auto connectivity(Mesh const &mesh) -> decltype(at_key<K>(mesh)) {
                return at_key<K>(mesh);
            }

            template <class Key,
                class Mesh,
                // rename to meta::list avoids requiring keys to be defined if incoming list is, e.g., std::tuple
                class K = meta::rename<meta::list, Key>,
                class Res = decltype(mesh_connectivity(K(), std::declval<Mesh const &>())),
                std::enable_if_t<!std::is_same<Res, not_provided>::value, int> = 0>
            Res connectivity(Mesh const &mesh) {
                return mesh_connectivity(K(), mesh);
            }
        } // namespace mesh

    } // namespace next
} // namespace gridtools
