/*cppimport
<%
cfg['compiler_args'] = ['-std=c++17', '-fopenmp', '-O0']
cfg['linker_args'] = [
    '-L/usr/local/lib/',
    '-fopenmp'
    ]
cfg['include_dirs'] = [
    '/workspace/eve_toolchain/examples/unstructured/fvm/dawn'
    ]
cfg['libraries'] = ['eckit', 'atlas']

setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>

#include "dawn/interface/atlas_interface.hpp"
#include "generated_fvm_nabla.hpp"

PYBIND11_MODULE(fvm_nabla_wrapper, m) {
  m.def("run_computation", [](atlas::Mesh &mesh, int ksize, atlas::Field S_MXX,
                              atlas::Field S_MYY, atlas::Field zavgS_MXX,
                              atlas::Field zavgS_MYY, atlas::Field pp,
                              atlas::Field pnabla_MXX, atlas::Field pnabla_MYY,
                              atlas::Field vol, atlas::Field sign) {
    auto S_MXX_view = atlasInterface::Field<double>(
        atlas::array::make_view<double, 2>(S_MXX));
    auto S_MYY_view = atlasInterface::Field<double>(
        atlas::array::make_view<double, 2>(S_MYY));
    auto zavgS_MXX_view = atlasInterface::Field<double>(
        atlas::array::make_view<double, 2>(zavgS_MXX));
    auto zavgS_MYY_view = atlasInterface::Field<double>(
        atlas::array::make_view<double, 2>(zavgS_MYY));
    auto pp_view =
        atlasInterface::Field<double>(atlas::array::make_view<double, 2>(pp));
    auto pnabla_MXX_view = atlasInterface::Field<double>(
        atlas::array::make_view<double, 2>(pnabla_MXX));
    auto pnabla_MYY_view = atlasInterface::Field<double>(
        atlas::array::make_view<double, 2>(pnabla_MYY));
    auto vol_view =
        atlasInterface::Field<double>(atlas::array::make_view<double, 2>(vol));
    auto sign_view = atlasInterface::SparseDimension<double>(
        atlas::array::make_view<double, 3>(sign));
    atlasInterface::Mesh interface_mesh{mesh};
    dawn_generated::cxxnaiveico::generated<atlasInterface::atlasTag>(
        interface_mesh, ksize, S_MXX_view, S_MYY_view, zavgS_MXX_view,
        zavgS_MYY_view, pp_view, pnabla_MXX_view, pnabla_MYY_view, vol_view,
        sign_view)
        .run();
  });
}
