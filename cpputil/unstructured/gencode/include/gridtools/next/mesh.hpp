struct vertex;
struct edge;
struct cell;

struct neighbor;

namespace gridtools {
namespace next {
namespace mesh {

template <class Connectivity>
auto get_neighbor_table(Connectivity const &connectivity) {
  return connectivity_get_neighbor_table(connectivity);
};

template <class Connectivity>
auto get_max_neighbors(Connectivity const &connectivity) {
  return connectivity_get_max_neighbors(connectivity);
}
} // namespace mesh
} // namespace next
} // namespace gridtools
