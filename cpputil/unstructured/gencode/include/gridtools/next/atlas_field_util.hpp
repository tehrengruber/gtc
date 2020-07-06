#include <cassert>

#include <atlas/field/Field.h>
#include <atlas/functionspace.h>
#include <gridtools/common/hymap.hpp>
#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/sid/delegate.hpp>

#include "unstructured.hpp"

#pragma once

// TODO this needs to live in GT or the SID converter needs to be implemented
// differently

namespace gridtools {
namespace sid {
namespace rename_all_dimensions_impl_ {
template <class NewHymapKeys, class Map> auto remap(Map map) {
  return hymap::convert_to<hymap::keys, NewHymapKeys>(std::move(map));
}

template <class NewHymapKeys, class Sid> struct renamed_sid : delegate<Sid> {
  template <class Map>
  using remapped_t = decltype(remap<NewHymapKeys>(std::declval<Map>()));

  template <class T>
  renamed_sid(T &&obj) : delegate<Sid>(std::forward<T>(obj)) {}

  friend remapped_t<strides_type<Sid>> sid_get_strides(renamed_sid const &obj) {
    return remap<NewHymapKeys>(sid_get_strides(obj.impl()));
  }
  friend remapped_t<lower_bounds_type<Sid>>
  sid_get_lower_bounds(renamed_sid const &obj) {
    return remap<NewHymapKeys>(sid_get_lower_bounds(obj.impl()));
  }
  friend remapped_t<upper_bounds_type<Sid>>
  sid_get_upper_bounds(renamed_sid const &obj) {
    return remap<NewHymapKeys>(sid_get_upper_bounds(obj.impl()));
  }
};

template <class...> struct stride_kind_wrapper {};

template <class NewHymapKeys, class Sid>
stride_kind_wrapper<NewHymapKeys, strides_kind<Sid>>
sid_get_strides_kind(renamed_sid<NewHymapKeys, Sid> const &);

template <class NewHymapKeys, class Sid>
renamed_sid<NewHymapKeys, Sid> rename_all_dimensions(Sid &&sid) {
  return renamed_sid<NewHymapKeys, Sid>{std::forward<Sid>(sid)};
}
} // namespace rename_all_dimensions_impl_
using rename_all_dimensions_impl_::rename_all_dimensions;
} // namespace sid
} // namespace gridtools

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
  // 2. k
  // 3. other dimensions
  // from this we can derive some checks:
  // - type of unstructured dimension (Done)
  // - no unstructured dimension tag after first dimension
  // - dim::k only in second (have unstructured dim) or first (no unstructured)
  // dimension
  template <class DataType> struct with_type {
    auto operator()(atlas::Field &field) {
      assert(field.rank() == sizeof...(DimensionTags) && "Rank mismatch");

      if constexpr (std::is_same<first_t, edge>{}) {
        if (field.functionspace().type().compare("Edges") != 0)
          atlas::throw_Exception("Not an edge field", Here());
      } else if constexpr (std::is_same<first_t, cell>{}) {
        if (field.functionspace().type().compare("Cells") != 0)
          atlas::throw_Exception("Not a cell field", Here());
      } else if constexpr (std::is_same<first_t, vertex>{}) {
        if (field.functionspace().type().compare("Nodes") != 0)
          atlas::throw_Exception("Not a vertex field", Here());
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
