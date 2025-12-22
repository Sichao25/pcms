
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pcms {

template<typename T, typename CoordinateElementType>
void bind_xgc_field_adapter(py::module& m, const std::string& type_suffix);

void bind_omega_h_field2(py::module& m);

void bind_transfer_field2_module(py::module& m);

void bind_omega_h_mesh_module(py::module& m);

void bind_interpolation_base_module(py::module& m);

void bind_coordinate_system_module(py::module& m);

void bind_coordinate_module(py::module& m);

void bind_create_field_module(py::module& m);

void bind_omega_h_field_layout_module(py::module& m);

}
PYBIND11_MODULE(py_pcms, m) {
  // Bind fundamental types first (coordinate systems, etc.)
  pcms::bind_coordinate_system_module(m);
  pcms::bind_coordinate_module(m);
  
  // Bind mesh and field infrastructure
  pcms::bind_omega_h_mesh_module(m);
  pcms::bind_omega_h_field_layout_module(m);
  pcms::bind_create_field_module(m);
  
  // Bind field types
  pcms::bind_xgc_field_adapter<double, double>(m, "double");
  pcms::bind_omega_h_field2(m);
  
  // Bind field operations
  pcms::bind_transfer_field2_module(m);
  pcms::bind_interpolation_base_module(m);
}