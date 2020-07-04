#include <array_fwd.h>
#include <atlas/array.h>
#include <atlas/grid.h>
#include <atlas/grid/StructuredGrid.h>
#include <atlas/mesh.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/meshgenerator.h>
#include <atlas/option.h>
#include <field/Field.h>
#include <functionspace/EdgeColumns.h>

#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/sid/concept.hpp>

auto make_mesh() {
  atlas::StructuredGrid structuredGrid = atlas::Grid("O2");
  atlas::MeshGenerator::Parameters generatorParams;
  generatorParams.set("triangulate", true);
  generatorParams.set("angle", -1.0);

  atlas::StructuredMeshGenerator generator(generatorParams);

  return generator.generate(structuredGrid);
}

int main() {
  auto mesh = make_mesh();
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

  static_assert(gridtools::is_sid<decltype(view)>{});
}
