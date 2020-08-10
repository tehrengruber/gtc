/**
 * Simple 3x3 Cartesian, periodic, hand-made mesh.
 *
 *      0e    1e    2e
 *   | ---- | ---- | ---- |
 * 9e|0v 10e|1v 11e|2v  9e|0v
 *   |  0c  |  1c  |  2c  |
 *   |  3e  |  4e  |  5e  |
 *   | ---- | ---- | ---- |
 *12e|3v 13e|4v 14e|5v 12e|3v
 *   |  3c  |  4c  |  5c  |
 *   |  6e  |  7e  |  8e  |
 *   | ---- | ---- | ---- |
 *15e|6v 16e|7v 17e|8v 15e| 6v
 *   |  6c  |  7c  |  8c  |
 *   |  0e  |  1e  |  2e  |
 *   | ---- | ---- | ---- |
 *    0v     1v     2v     0v
 *
 */

#include "../mesh.hpp"
#include "../unstructured.hpp"
#include <cstddef>
#include <gridtools/storage/builder.hpp>
#include <type_traits>

#ifdef __CUDACC__ // TODO proper handling
#include <gridtools/storage/gpu.hpp>
using storage_trait = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_trait = gridtools::storage::cpu_ifirst;
#endif

namespace gridtools {
    namespace next {
        namespace test_helper {
            struct simple_mesh {
                template <class LocationType>
                struct primary_connectivity {
                    std::size_t size_;

                    GT_FUNCTION friend std::size_t connectivity_size(primary_connectivity const &conn) {
                        return conn.size_;
                    }
                };

                template <class LocationType, std::size_t Size, std::size_t Neighbors>
                struct regular_connectivity {
                    struct builder {
                        auto operator()() {
                            return gridtools::storage::builder<storage_trait>.template type<int>().template layout<0, 1>().template dimensions(
                    Size, std::integral_constant<std::size_t, Neighbors>{});
                        }
                    };

                    decltype(builder{}()()) tbl_;
                    static constexpr gridtools::int_t missing_value_ = -1;

                    regular_connectivity(std::array<std::array<int, Neighbors>, Size>) {}
                };

                template <class Key, std::enable_if_t<std::is_same_v<Key, vertex>, int> = 0>
                decltype(auto) mesh_connectivity(const simple_mesh &mesh) {
                    return primary_connectivity<vertex>{9};
                }
                template <class Key, std::enable_if_t<std::is_same_v<Key, edge>, int> = 0>
                decltype(auto) mesh_connectivity(const simple_mesh &mesh) {
                    return primary_connectivity<edge>{18};
                }
                template <class Key, std::enable_if_t<std::is_same_v<Key, cell>, int> = 0>
                decltype(auto) mesh_connectivity(const simple_mesh &mesh) {
                    return primary_connectivity<cell>{9};
                }
            };
        } // namespace test_helper
    }     // namespace next
} // namespace gridtools

/**
 * TODO maybe later: Simple 2x2 Cartesian hand-made mesh with one halo line.
 *
 *     0e    1e    2e    3e
 *   | --- | --- | --- | --- |
 *   |0v   |1v   |2v   |3v   |4v
 *   |  0c |  1c |  2c |  3c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   |  4c |  5c |  6c |  7c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   |  8c |  9c | 10c | 11c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   | 12c | 13c | 14c | 15c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *
 *
 */
