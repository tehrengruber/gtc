#include <gridtools/next/test_helper/simple_mesh.hpp>

#include <gtest/gtest.h>

TEST(simple_mesh, test) {
    gridtools::next::test_helper::simple_mesh::regular_connectivity<edge, 2, 2>({{{1, 2}, {3, 4}}});
}
