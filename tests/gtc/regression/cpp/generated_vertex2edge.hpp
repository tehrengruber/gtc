#include <gridtools/common/cuda_util.hpp>
#include <gridtools/next/cuda_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/tmp_gpu_storage.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/composite.hpp>

namespace sten_impl_ {
struct connectivity_tag;
struct field_in_tag;
struct field_out_tag;

template <class edge_connectivity_t, class edge2vertex_connectivity_t,
          class edge_origins_t, class edge_strides_t,
          class edge_vertex_origins_t, class edge_vertex_strides_t>
__global__ void
kernelHorizontalLoop_45(edge_connectivity_t edge_connectivity,
                        edge2vertex_connectivity_t edge2vertex_connectivity,
                        edge_origins_t edge_origins,
                        edge_strides_t edge_strides,
                        edge_vertex_origins_t edge_vertex_origins,
                        edge_vertex_strides_t edge_vertex_strides) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= gridtools::next::connectivity::size(edge_connectivity))
    return;

  auto edge_ptrs = edge_origins();
  gridtools::sid::shift(edge_ptrs,
                        gridtools::device::at_key<edge>(edge_strides), idx);

  double localNeighborReduce_12 = (double)0.0;
  localNeighborReduce_12 = (double)0;
  for (int neigh = 0; neigh < gridtools::next::connectivity::max_neighbors(
                                  edge2vertex_connectivity);
       ++neigh) {
    // body
    auto absolute_neigh_index =
        *gridtools::device::at_key<connectivity_tag>(edge_ptrs);
    if (absolute_neigh_index !=
        gridtools::next::connectivity::skip_value(edge2vertex_connectivity)) {
      auto edge_vertex_ptrs = edge_vertex_origins();
      gridtools::sid::shift(
          edge_vertex_ptrs,
          gridtools::device::at_key<vertex>(edge_vertex_strides),
          absolute_neigh_index);

      localNeighborReduce_12 =
          (localNeighborReduce_12 +
           *gridtools::device::at_key<field_in_tag>(edge_vertex_ptrs));
      // body end
    }
    gridtools::sid::shift(edge_ptrs,
                          gridtools::device::at_key<neighbor>(edge_strides), 1);
  }
  gridtools::sid::shift(
      edge_ptrs, gridtools::device::at_key<neighbor>(edge_strides),
      -gridtools::next::connectivity::max_neighbors(edge2vertex_connectivity));
  *gridtools::device::at_key<field_out_tag>(edge_ptrs) = localNeighborReduce_12;
}

} // namespace sten_impl_

template <class mesh_t, class field_in_t, class field_out_t>
void sten(mesh_t &&mesh, field_in_t &&field_in, field_out_t &&field_out) {
  namespace tu = gridtools::tuple_util;
  using namespace sten_impl_;

  {
    auto primary_connectivity = gridtools::next::mesh::connectivity<edge>(mesh);
    auto edge2vertex_connectivity =
        gridtools::next::mesh::connectivity<std::tuple<edge, vertex>>(mesh);

    auto edge_fields =
        tu::make<gridtools::sid::composite::keys<field_out_tag,
                                                 connectivity_tag>::values>(
            field_out, gridtools::next::connectivity::neighbor_table(
                           edge2vertex_connectivity));
    static_assert(gridtools::is_sid<decltype(edge_fields)>{});

    auto edge_vertex_fields =
        tu::make<gridtools::sid::composite::keys<field_in_tag>::values>(
            field_in);
    static_assert(gridtools::is_sid<decltype(edge_vertex_fields)>{});

    auto [blocks, threads_per_block] = gridtools::next::cuda_util::cuda_setup(
        gridtools::next::connectivity::size(primary_connectivity));
    kernelHorizontalLoop_45<<<blocks, threads_per_block>>>(
        primary_connectivity, edge2vertex_connectivity,
        gridtools::sid::get_origin(edge_fields),
        gridtools::sid::get_strides(edge_fields),
        gridtools::sid::get_origin(edge_vertex_fields),
        gridtools::sid::get_strides(edge_vertex_fields));
    GT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}
