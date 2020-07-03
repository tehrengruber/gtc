#pragma once

#include <gridtools/common/hymap.hpp>

namespace gridtools {
namespace next {
namespace mesh {

// models hymaps as Mesh
// TOOD(havogt): only if values model the Connectivity concept
template <typename Key, typename Hymap> // TODO protect
decltype(auto) mesh_connectivity(const Hymap &mesh) {
  return at_key<Key>(mesh);
}

} // namespace mesh
} // namespace next
} // namespace gridtools
