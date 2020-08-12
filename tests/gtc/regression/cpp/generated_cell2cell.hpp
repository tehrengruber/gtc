#include <gridtools/common/cuda_util.hpp>
#include <gridtools/next/cuda_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/tmp_gpu_storage.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/composite.hpp>

namespace sten_impl_ {
struct connectivity_tag;
struct field_out_tag;
struct field_in_tag;

template <class cell_connectivity_t, class cell2cell_connectivity_t,
          class cell_origins_t, class cell_strides_t, class cell_cell_origins_t,
          class cell_cell_strides_t>
__global__ void
kernelHorizontalLoop_50(cell_connectivity_t cell_connectivity,
                        cell2cell_connectivity_t cell2cell_connectivity,
                        cell_origins_t cell_origins,
                        cell_strides_t cell_strides,
                        cell_cell_origins_t cell_cell_origins,
                        cell_cell_strides_t cell_cell_strides) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= gridtools::next::connectivity::size(cell_connectivity))
    return;

  auto cell_ptrs = cell_origins();
  gridtools::sid::shift(cell_ptrs,
                        gridtools::device::at_key<cell>(cell_strides), idx);

  double localNeighborReduce_15 = (double)0.0;
  localNeighborReduce_15 = (double)0;
  for (int neigh = 0; neigh < gridtools::next::connectivity::max_neighbors(
                                  cell2cell_connectivity);
       ++neigh) {
    // body
    auto absolute_neigh_index =
        *gridtools::device::at_key<connectivity_tag>(cell_ptrs);
    if (absolute_neigh_index !=
        gridtools::next::connectivity::skip_value(cell2cell_connectivity)) {
      auto cell_cell_ptrs = cell_cell_origins();
      gridtools::sid::shift(cell_cell_ptrs,
                            gridtools::device::at_key<cell>(cell_cell_strides),
                            absolute_neigh_index);

      localNeighborReduce_15 =
          (localNeighborReduce_15 +
           (*gridtools::device::at_key<field_in_tag>(cell_ptrs) +
            *gridtools::device::at_key<field_in_tag>(cell_cell_ptrs)));
      // body end
    }
    gridtools::sid::shift(cell_ptrs,
                          gridtools::device::at_key<neighbor>(cell_strides), 1);
  }
  gridtools::sid::shift(
      cell_ptrs, gridtools::device::at_key<neighbor>(cell_strides),
      -gridtools::next::connectivity::max_neighbors(cell2cell_connectivity));
  *gridtools::device::at_key<field_out_tag>(cell_ptrs) = localNeighborReduce_15;
}

} // namespace sten_impl_

template <class mesh_t, class field_in_t, class field_out_t>
void sten(mesh_t &&mesh, field_in_t &&field_in, field_out_t &&field_out) {
  namespace tu = gridtools::tuple_util;
  using namespace sten_impl_;

  {
    auto primary_connectivity = gridtools::next::mesh::connectivity<cell>(mesh);
    auto cell2cell_connectivity =
        gridtools::next::mesh::connectivity<std::tuple<cell, cell>>(mesh);

    auto cell_fields = tu::make<gridtools::sid::composite::keys<
        field_in_tag, field_out_tag, connectivity_tag>::values>(
        field_in, field_out,
        gridtools::next::connectivity::neighbor_table(cell2cell_connectivity));
    static_assert(gridtools::is_sid<decltype(cell_fields)>{});

    auto cell_cell_fields =
        tu::make<gridtools::sid::composite::keys<field_in_tag>::values>(
            field_in);
    static_assert(gridtools::is_sid<decltype(cell_cell_fields)>{});

    auto [blocks, threads_per_block] = gridtools::next::cuda_util::cuda_setup(
        gridtools::next::connectivity::size(primary_connectivity));
    kernelHorizontalLoop_50<<<blocks, threads_per_block>>>(
        primary_connectivity, cell2cell_connectivity,
        gridtools::sid::get_origin(cell_fields),
        gridtools::sid::get_strides(cell_fields),
        gridtools::sid::get_origin(cell_cell_fields),
        gridtools::sid::get_strides(cell_cell_fields));
    GT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}
