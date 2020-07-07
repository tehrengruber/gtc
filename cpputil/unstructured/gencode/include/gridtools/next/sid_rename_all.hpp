#pragma once

#include <gridtools/common/hymap.hpp>
#include <gridtools/sid/delegate.hpp>

// TODO this needs to live in GT or the SID converter needs to be implemented
// differently

namespace gridtools {
namespace sid {
namespace rename_all_dimensions_impl_ {
template <class NewHymapKeys, class Map> auto remap(Map map) {
  return hymap::convert_to<hymap::keys, NewHymapKeys>(std::move(map));
}

template <class NewHymapKeys, class Sid>
struct renamed_all_sid : delegate<Sid> {
  using delegate<Sid>::delegate;
};

template <class...> struct stride_kind_wrapper {};

template <class NewHymapKeys, class Sid>
stride_kind_wrapper<NewHymapKeys,
                    decltype(sid_get_strides_kind(std::declval<Sid const &>()))>
sid_get_strides_kind(renamed_all_sid<NewHymapKeys, Sid> const &);

template <class NewHymapKeys, class Sid>
decltype(remap<NewHymapKeys>(sid_get_strides(std::declval<Sid const &>())))
sid_get_strides(renamed_all_sid<NewHymapKeys, Sid> const &obj) {
  return remap<NewHymapKeys>(sid_get_strides(obj.m_impl));
}

template <class NewHymapKeys, class Sid>
decltype(remap<NewHymapKeys>(sid_get_lower_bounds(std::declval<Sid const &>())))
sid_get_lower_bounds(renamed_all_sid<NewHymapKeys, Sid> const &obj) {
  return remap<NewHymapKeys>(sid_get_lower_bounds(obj.m_impl));
}

template <class NewHymapKeys, class Sid>
decltype(remap<NewHymapKeys>(sid_get_upper_bounds(std::declval<Sid const &>())))
sid_get_upper_bounds(renamed_all_sid<NewHymapKeys, Sid> const &obj) {
  return remap<NewHymapKeys>(sid_get_upper_bounds(obj.m_impl));
}

template <class NewHymapKeys, class Sid>
renamed_all_sid<NewHymapKeys, Sid> rename_all_dimensions(Sid &&sid) {
  return renamed_all_sid<NewHymapKeys, Sid>{std::forward<Sid>(sid)};
}
} // namespace rename_all_dimensions_impl_
using rename_all_dimensions_impl_::rename_all_dimensions;
} // namespace sid
} // namespace gridtools

// namespace gridtools {
// namespace sid {
// namespace rename_all_dimensions_impl_ {
// template <class NewHymapKeys, class Map> auto remap(Map map) {
//   return hymap::convert_to<hymap::keys, NewHymapKeys>(std::move(map));
// }

// template <class NewHymapKeys, class Sid> struct renamed_all_sid :
// delegate<Sid> {
//   template <class T>
//   renamed_all_sid(T &&obj) : delegate<Sid>(std::forward<T>(obj)) {}
// };

// template <class...> struct stride_kind_wrapper {};

// template <class NewHymapKeys, class Sid>
// stride_kind_wrapper<NewHymapKeys, strides_kind<Sid>>
// sid_get_strides_kind(renamed_all_sid<NewHymapKeys, Sid> const &);

// template <class NewHymapKeys, class Sid>
// renamed_all_sid<NewHymapKeys, Sid> rename_all_dimensions(Sid &&sid) {
//   return renamed_all_sid<NewHymapKeys, Sid>{std::forward<Sid>(sid)};
// }

// template <class NewHymapKeys, class Sid>
// decltype(remap<NewHymapKeys>(sid_get_lower_bounds(std::declval<Sid const
// &>()))) sid_get_lower_bounds(renamed_all_sid<NewHymapKeys, Sid> const &obj) {
//   return remap<NewHymapKeys>(sid_get_lower_bounds(obj.impl()));
// }

// template <class NewHymapKeys, class Sid>
// decltype(remap<NewHymapKeys>(sid_get_upper_bounds(std::declval<Sid const
// &>()))) sid_get_upper_bounds(renamed_all_sid<NewHymapKeys, Sid> const &obj) {
//   return remap<NewHymapKeys>(sid_get_upper_bounds(obj.impl()));
// }
// } // namespace rename_all_dimensions_impl_
// using rename_all_dimensions_impl_::rename_all_dimensions;
// } // namespace sid
// } // namespace gridtools
