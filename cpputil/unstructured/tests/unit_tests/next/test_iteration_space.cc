#include <gridtools/common/tuple_util.hpp>
#include <gridtools/next/iteration_space.hpp>
#include <gridtools/sid/concept.hpp>
#include <gtest/gtest.h>

using namespace gridtools::next;

namespace {
    struct dummy;

    TEST(regular_iteration_space, is_sid) {
        using namespace gridtools;
        // using namespace gridtools::sid::concept_impl_;

        using Sid = regular_iteration_space<dummy>;

        // using PtrHolder = ptr_holder_type<Sid>;
        // using Ptr = ptr_type<Sid>;
        // using ReferenceType = reference_type<Sid>;
        // using PtrDiff = ptr_diff_type<Sid>;
        // using StridesType = strides_type<Sid>;
        // using StrideTypeList = gridtools::tuple_util::traits::to_types<std::decay_t<StridesType>>;
        // using StridesKind = strides_kind<Sid>;
        // using LowerBoundsType = lower_bounds_type<Sid>;
        // using UpperBoundsType = upper_bounds_type<Sid>;

        // static_assert(std::is_trivially_copyable<PtrHolder>{});
        // static_assert(std::is_trivially_copyable<StridesType>{});
        // static_assert(std::is_default_constructible<PtrDiff>{});
        // static_assert(
        //     std::is_convertible<decltype(std::declval<Ptr const &>() + std::declval<PtrDiff const &>()), Ptr>{});

        static_assert(gridtools::is_sid<Sid>{});

        constexpr int start = 1;
        constexpr int end = 10;

        regular_iteration_space<dummy> testee{start, end};
        auto origin = sid::get_origin(testee);
        ASSERT_EQ(start, *origin);

        auto lower_bounds = sid::get_lower_bounds(testee);
        ASSERT_EQ(start, sid::get_lower_bound<dummy>(lower_bounds));

        auto upper_bounds = sid::get_upper_bounds(testee);
        ASSERT_EQ(end, sid::get_upper_bound<dummy>(upper_bounds));

        // helper
        ASSERT_EQ(end - start, sid_helper::size<dummy>(testee));
    }
} // namespace
