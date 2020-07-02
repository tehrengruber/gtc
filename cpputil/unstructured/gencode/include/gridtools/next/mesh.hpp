#include <cstddef>
#include <gridtools/common/hymap.hpp>
struct vertex;
struct edge;
struct cell;

struct neighbor;

namespace gridtools {
namespace next {

namespace connectivity {
template <class Connectivity>
auto get_neighbor_table(Connectivity const &connectivity) {
  return connectivity_get_neighbor_table(connectivity);
};

template <class Connectivity>
auto get_max_neighbors(Connectivity const &connectivity) {
  return connectivity_get_max_neighbors(connectivity);
}

template <class Connectivity>
std::size_t get_primary_size(Connectivity const &connectivity) {
  return connectivity_get_primary_size(connectivity);
}
} // namespace connectivity

namespace mesh {
template <typename Key, typename Mesh>
decltype(auto) mesh_get_connectivity(const Mesh &mesh);

template <class Key, class Mesh>
decltype(auto) get_connectivity(Mesh const &mesh) {
  return mesh_get_connectivity<Key>(mesh);
};
} // namespace mesh

} // namespace next
} // namespace gridtools
