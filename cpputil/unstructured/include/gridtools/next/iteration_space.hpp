#pragma once

#include <gridtools/common/host_device.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/sid/concept.hpp>
#include <type_traits>

namespace gridtools::next {
    // similar to gridtools::stencil::positional

    struct range_iteration_space_stride {};

    // TODO probably the LocationType should be removed everywhere
    template <class LocationType>
    struct range_iteration_space {
        const int m_from; // TODO or integral_constant
        const int m_to;
        int m_cur;

        GT_FUNCTION range_iteration_space(int from, int to, int cur) : m_from{from}, m_to{to}, m_cur{cur} {};
        GT_FUNCTION range_iteration_space(int from, int to) : m_from{from}, m_to{to}, m_cur{from} {};
        GT_FUNCTION int operator*() const { return m_cur; }
        GT_FUNCTION range_iteration_space const &operator()() const { return *this; }
    };

    template <class LocationType>
    GT_FUNCTION range_iteration_space<LocationType> operator+(range_iteration_space<LocationType> lhs, int rhs) {
        return {lhs.m_from, lhs.m_to, lhs.m_cur + rhs};
    }

    template <class LocationType>
    typename hymap::keys<LocationType>::template values<range_iteration_space_stride> sid_get_strides(
        range_iteration_space<LocationType>) {
        return {};
    }

    template <class LocationType>
    typename hymap::keys<LocationType>::template values<int> sid_get_lower_bounds(
        range_iteration_space<LocationType> s) {
        return {s.m_from};
    }

    template <class LocationType>
    typename hymap::keys<LocationType>::template values<int> sid_get_upper_bounds(
        range_iteration_space<LocationType> s) {
        return {s.m_to};
    }

    template <class LocationType>
    GT_FUNCTION void sid_shift(range_iteration_space<LocationType> &p, range_iteration_space_stride, int_t offset) {
        p.m_cur += offset;
    }

    GT_FUNCTION void sid_shift(int &ptr_diff, range_iteration_space_stride, int_t offset) { ptr_diff += offset; }

    template <class LocationType>
    int sid_get_ptr_diff(range_iteration_space<LocationType>);

    template <class LocationType>
    range_iteration_space<LocationType> sid_get_origin(range_iteration_space<LocationType> obj) {
        return obj;
    }

    namespace sid_helper {
        // TODO maybe add to sid as generic helper?
        template <class Key, class Sid>
        std::enable_if_t<is_sid<Sid>{}, int> size(Sid &&sid) {
            return gridtools::sid::get_upper_bound<Key>(gridtools::sid::get_upper_bounds(std::forward<Sid>(sid))) -
                   gridtools::sid::get_lower_bound<Key>(gridtools::sid::get_lower_bounds(std::forward<Sid>(sid)));
        }

        template <class Key, class Sid>
        std::enable_if_t<is_sid<Sid>{}, int> lower_bound(Sid &&sid) {
            return gridtools::at_key<Key>(gridtools::sid::get_lower_bounds(std::forward<Sid>(sid)));
        }
        template <class Key, class Sid>
        std::enable_if_t<is_sid<Sid>{}, int> upper_bound(Sid &&sid) {
            return gridtools::at_key<Key>(gridtools::sid::get_upper_bounds(std::forward<Sid>(sid)));
        }

    } // namespace sid_helper
} // namespace gridtools::next
