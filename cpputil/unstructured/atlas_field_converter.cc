#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildDualMesh.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"
#include "gridtools/common/integral_constant.hpp"
#include <array_fwd.h>
#include <atlas/array.h>
#include <atlas/grid/StructuredGrid.h>
#include <atlas/mesh.h>
#include <atlas/option.h>
#include <field/Field.h>
#include <functionspace/EdgeColumns.h>
#include <type_traits>

#include "gridtools/next/atlas_array_view_adapter.hpp"
#include <gridtools/next/atlas_adapter.hpp>
#include <gridtools/next/atlas_field_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/sid/synthetic.hpp>

#include "gridtools/sid/rename_dimension.hpp"
#include "tests/include/util/atlas_util.hpp"

namespace dim {
struct k;
} // namespace dim

// field to sid converter

// template <class LocationType, class DataType> auto as(atlas::Field &field) {
//   assert(field.rank() == 2 && "only rank 2 can be converted to "
//                               "uSID");

//   if constexpr (std::is_same<LocationType, edge>{}) {
//     assert(field.functionspace().type().compare("Edges") == 0 &&
//            "wrong location type");
//   } else if constexpr (std::is_same<LocationType, cell>{}) {
//     assert(field.functionspace().type().compare("Cells") == 0 &&
//            "wrong location type");
//   } else if constexpr (std::is_same<LocationType, vertex>{}) {
//     assert(field.functionspace().type().compare("Nodes") == 0 &&
//            "wrong location type");
//   }
//   using strides_t =
//       typename gridtools::hymap::keys<LocationType,
//                                       dim::k>::template values<int, int>;

//   return gridtools::sid::synthetic()
//       .set<gridtools::sid::property::origin>(
//           gridtools::sid::make_simple_ptr_holder(
//               &atlas::array::make_view<DataType, 2>(field)(0, 0)))
//       .template set<gridtools::sid::property::strides>(
//           strides_t(field.strides()[0], field.strides()[1]))
//       .template set<gridtools::sid::property::strides_kind, strides_t>();
// }

int main() {
  auto mesh = atlas_util::make_mesh();
  atlas::mesh::actions::build_edges(mesh);

  int nb_levels = 5;
  atlas::functionspace::EdgeColumns fs_edges(
      mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

  atlas::Field f;
  auto my_field = fs_edges.createField<double>(atlas::option::name("my_field"));

  auto view = atlas::array::make_view<double, 2>(my_field);
  for (int i = 0; i < fs_edges.size(); ++i)
    for (int k = 0; k < nb_levels; ++k)
      view(i, k) = i * 10 + k;

  //   auto my_field_sidified = as<edge, double>(my_field);
  //   static_assert(gridtools::is_sid<decltype(my_field_sidified)>{});

  auto my_field_sidified =
      gridtools::next::atlas_util::as<edge, dim::k>::with_type<double>{}(
          my_field);
  static_assert(gridtools::is_sid<decltype(my_field_sidified)>{});

  auto strides = gridtools::sid::get_strides(my_field_sidified);
  for (int i = 0; i < fs_edges.size(); ++i)
    for (int k = 0; k < nb_levels; ++k) {
      auto ptr = gridtools::sid::get_origin(my_field_sidified)();
      gridtools::sid::shift(ptr, gridtools::at_key<edge>(strides), i);
      gridtools::sid::shift(ptr, gridtools::at_key<dim::k>(strides), k);
      std::cout << view(i, k) << "/" << *ptr << std::endl;
    }
}
