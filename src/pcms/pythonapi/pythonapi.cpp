
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pcms
{

void bind_omega_h_field2(py::module& m);

void bind_transfer_field2_module(py::module& m);

void bind_omega_h_mesh_module(py::module& m);

void bind_coordinate_system_module(py::module& m);

void bind_coordinate_module(py::module& m);

void bind_create_field_module(py::module& m);

void bind_field_layout_module(py::module& m);

void bind_field_module(py::module& m);

void bind_omega_h_field_layout_module(py::module& m);

void bind_uniform_grid_field_layout_module(py::module& m);

void bind_uniform_grid_field_module(py::module& m);

void bind_mls_interpolation_module(py::module& m);

void bind_mesh_utilities_module(py::module& m);

} // namespace pcms
PYBIND11_MODULE(pcms, m)
{
  // Bind fundamental types first (coordinate systems, etc.)
  pcms::bind_coordinate_system_module(m);
  pcms::bind_coordinate_module(m);

  // Bind mesh and field infrastructure
  pcms::bind_omega_h_mesh_module(m);
  pcms::bind_field_layout_module(m);
  // bind_field_module is a no-op stub — FieldT<T>/LocalizationHint/FieldDataView
  // have been removed from the C++ API.
  pcms::bind_field_module(m);
  pcms::bind_omega_h_field_layout_module(m);
  pcms::bind_uniform_grid_field_layout_module(m);
  // bind_create_field_module registers LagrangeFunctionSpace and FieldData<Real>
  pcms::bind_create_field_module(m);

  // Bind OutOfBoundsPolicy and helpers (bind_omega_h_field2 is now a minimal
  // module that only exposes OutOfBoundsMode/OutOfBoundsPolicy + coordinate helpers)
  pcms::bind_omega_h_field2(m);
  // bind_uniform_grid_field_module is a no-op stub — UniformGridField<N> has
  // been removed; use LagrangeFunctionSpace::from_uniform_grid instead.
  pcms::bind_uniform_grid_field_module(m);

  // Bind field operations and Interpolator
  pcms::bind_transfer_field2_module(m);

  // Bind interpolator operations
  pcms::bind_mls_interpolation_module(m);

  // Bind mesh utility functions
  pcms::bind_mesh_utilities_module(m);
}