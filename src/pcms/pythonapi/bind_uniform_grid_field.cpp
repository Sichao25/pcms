#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pcms
{

// UniformGridField<N> (the old monolithic field type that combined layout,
// data storage, and evaluation) has been removed. Uniform-grid functionality
// is now available via:
//   - UniformGridFieldLayout<N>  (layout, bound in bind_uniform_grid_field_layout.cpp)
//   - LagrangeFunctionSpace::from_uniform_grid (factory, bound in bind_field_base.cpp)
//   - SimpleFieldData<Real> / FieldData<Real>  (data, bound in bind_field_base.cpp)
// This stub is kept so that the build system does not need to be modified.
void bind_uniform_grid_field_module(py::module& /*m*/) {}

} // namespace pcms
