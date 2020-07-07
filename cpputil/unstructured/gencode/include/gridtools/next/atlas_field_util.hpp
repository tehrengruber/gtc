#include <cassert>

#include <atlas/field/Field.h>
#include <atlas/functionspace.h>
#include <gridtools/common/hymap.hpp>
#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/next/sid_rename_all.hpp>
#include <gridtools/sid/delegate.hpp>

#include "unstructured.hpp"

#pragma once

namespace gridtools::next::atlas_util {

// THE COMMENTED IS A DIRECT SID ADAPTION, THE OTHER IS USING THE
// ARRAY_VIEW_ADAPTER AND RENAMES:
// using strides_t =
//     typename gridtools::hymap::keys<DimensionTags...>::template
//     values<int, int>;
//
// return gridtools::sid::synthetic()
//     .set<gridtools::sid::property::origin>(
//         gridtools::sid::make_simple_ptr_holder(
//             atlas::array::make_view<DataType, 2>(field).data()))
//     .template set<gridtools::sid::property::strides>(strides_t(
//         field.strides()[0], field.strides()[1])) // TODO more
//         dimensions
//     .template set<gridtools::sid::property::strides_kind, strides_t>();

template <class... DimensionTags> struct as {
  template <class First, class... Rest> struct first_impl {
    using type = First;
  };
  using first_t = typename first_impl<DimensionTags...>::type;

  // Atlas field dimensions order (TODO double check):
  // 1. unstructured dimension
  // 2. vertical
  // 3. other dimensions
  // from this we can derive some checks:
  // - runtime: type of unstructured dimension (Done)
  // - static: dim::k in first (no unstructured) or second dimension
  // - static: no unstructured dimension tag after first dimension
  template <class DataType> struct with_type {
    auto operator()(atlas::Field &field) {
      assert(field.rank() == sizeof...(DimensionTags) && "Rank mismatch");

      if constexpr (std::is_same<first_t, edge>{}) {
        if (field.functionspace().type().compare("EdgeColumns") != 0 &&
            field.functionspace().type().compare("Edges") !=
                0) // TODO EdgeColumns vs Edges
          atlas::throw_Exception(field.name() + " is not an edge field",
                                 Here());
      } else if constexpr (std::is_same<first_t, cell>{}) {
        if (field.functionspace().type().compare("Cells") != 0)
          atlas::throw_Exception(field.name() + " is not a cell field", Here());
      } else if constexpr (std::is_same<first_t, vertex>{}) {
        if (field.functionspace().type().compare("NodeColumns") !=
            0) // TODO NodeColumns vs Nodes
          atlas::throw_Exception(field.name() + " is not a vertex field",
                                 Here());
      }

      static_assert(
          gridtools::is_sid<decltype(
              atlas::array::make_view<DataType, sizeof...(DimensionTags)>(
                  field))>{},
          "ArrayView is not SID. Did you include "
          "\"atlas_array_view_adapter.hpp\"?");
      return gridtools::sid::rename_all_dimensions<
          gridtools::hymap::keys<DimensionTags...>>(
          atlas::array::make_view<DataType, sizeof...(DimensionTags)>(field));
    }
  };
};
} // namespace gridtools::next::atlas_util
