#pragma once

#include <gridtools/common/hymap.hpp>

namespace gridtools {
namespace next {
namespace mesh {

// models hymaps as Mesh
// TODO(havogt): protection: only if values model the Connectivity concept
template <typename Key, typename Hymap> // TODO protect, only hymaps allowed
decltype(auto) mesh_connectivity(const Hymap &mesh) {
  return at_key<Key>(mesh);
}

} // namespace mesh
} // namespace next
} // namespace gridtools
