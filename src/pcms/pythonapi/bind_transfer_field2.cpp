#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pcms/transfer_field2.h"
#include "pcms/field.h"

namespace py = pybind11;

namespace pcms {

template<typename T>
void bind_transfer_field2(py::module& m, const std::string& type_suffix) {
  std::string copy_name = "copy_field2_" + type_suffix;
  std::string interpolate_name = "interpolate_field2_" + type_suffix;

  m.def(copy_name.c_str(),
        [](const FieldT<T>& source, FieldT<T>& target) {
          copy_field2(source, target);
        },
        py::arg("source"),
        py::arg("target"),
        "Copy field data from source to target. Source and target must be of the same type.");

  m.def(interpolate_name.c_str(),
        [](const FieldT<T>& source, FieldT<T>& target) {
          interpolate_field2(source, target);
        },
        py::arg("source"),
        py::arg("target"),
        "Interpolate field from source to target. Coordinate systems must match.");
}

void bind_transfer_field2_module(py::module& m) {
  // Bind for Real type (typically double)
  bind_transfer_field2<Real>(m, "Real");

  // If Real is not double, also bind double explicitly
  if constexpr (!std::is_same_v<Real, double>) {
    bind_transfer_field2<double>(m, "Double");
  }

  // Bind for float if needed
  bind_transfer_field2<float>(m, "Float");

  // Add convenience aliases for the most common case (Real)
  m.def("interpolate_field",
        [](const FieldT<Real>& source, FieldT<Real>& target) {
          interpolate_field2(source, target);
        },
        py::arg("source"),
        py::arg("target"),
        "Interpolate field from source to target. Coordinate systems must match.");

  m.def("copy_field",
        [](const FieldT<Real>& source, FieldT<Real>& target) {
          copy_field2(source, target);
        },
        py::arg("source"),
        py::arg("target"),
        "Copy field data from source to target. Source and target must be of the same type.");
}

} // namespace pcms