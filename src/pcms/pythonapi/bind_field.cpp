#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pcms/field.h"
#include "helpers.h"

namespace py = pybind11;

namespace pcms {

template<typename T>
void bind_field_t(py::module& m, const std::string& type_suffix) {
  std::string class_name = "FieldT_" + type_suffix;
  
  py::class_<FieldT<T>, std::shared_ptr<FieldT<T>>>(m, class_name.c_str())
    .def("get_coordinate_system", &FieldT<T>::GetCoordinateSystem,
         "Get the coordinate system of the field")
    
    .def("get_localization_hint", &FieldT<T>::GetLocalizationHint,
         py::arg("coordinates"),
         "Get a localization hint for a set of coordinates")
    
    .def("evaluate", [](const FieldT<T>& self,
                       LocalizationHint hint,
                       py::array_t<T> results_array,
                       CoordinateSystem coord_sys) {
      auto results_view = numpy_to_view<T>(results_array);
      FieldDataView<T, HostMemorySpace> results(results_view, coord_sys);
      self.Evaluate(hint, results);
    },
    py::arg("hint"),
    py::arg("results"),
    py::arg("coordinate_system"),
    "Evaluate the field at given locations")
    
    .def("evaluate_gradient", [](FieldT<T>& self,
                                 py::array_t<T> results_array,
                                 CoordinateSystem coord_sys) {
      auto results_view = numpy_to_view<T>(results_array);
      FieldDataView<T, HostMemorySpace> results(results_view, coord_sys);
      self.EvaluateGradient(results);
    },
    py::arg("results"),
    py::arg("coordinate_system"),
    "Evaluate the gradient of the field")
    
    .def("get_dof_holder_data", [](const FieldT<T>& self) {
      auto data = self.GetDOFHolderData();
      return view_to_numpy(data);
    },
    "Get the DOF holder data")
    
    .def("set_dof_holder_data", [](FieldT<T>& self, py::array_t<const T> data) {
      auto data_view = numpy_to_view<const T>(data);
      self.SetDOFHolderData(data_view);
    },
    py::arg("data"),
    "Set the DOF holder data")
    
    .def("get_layout", &FieldT<T>::GetLayout,
         py::return_value_policy::reference,
         "Get the field layout")
    
    .def("can_evaluate_gradient", &FieldT<T>::CanEvaluateGradient,
         "Check if the field can evaluate gradients")
    
    .def("serialize", [](const FieldT<T>& self,
                        py::array_t<T> buffer,
                        py::array_t<const LO> permutation) {
      auto buffer_view = numpy_to_view<T>(buffer);
      auto perm_view = numpy_to_view<const LO>(permutation);
      return self.Serialize(buffer_view, perm_view);
    },
    py::arg("buffer"),
    py::arg("permutation"),
    "Serialize the field data")
    
    .def("deserialize", [](FieldT<T>& self,
                          py::array_t<const T> buffer,
                          py::array_t<const LO> permutation) {
      auto buffer_view = numpy_to_view<const T>(buffer);
      auto perm_view = numpy_to_view<const LO>(permutation);
      self.Deserialize(buffer_view, perm_view);
    },
    py::arg("buffer"),
    py::arg("permutation"),
    "Deserialize field data from buffer");
}

void bind_field_module(py::module& m) {
  // Bind LocalizationHint (opaque type - data member is internal only)
  py::class_<LocalizationHint>(m, "LocalizationHint")
    .def(py::init<>());
  
  // Bind FieldDataView for common types
  py::class_<FieldDataView<Real, HostMemorySpace>>(m, "FieldDataView_Real")
    .def(py::init<Rank1View<Real, HostMemorySpace>, CoordinateSystem>(),
         py::arg("values"),
         py::arg("coordinate_system"))
    .def("size", &FieldDataView<Real, HostMemorySpace>::Size)
    .def("get_coordinate_system", &FieldDataView<Real, HostMemorySpace>::GetCoordinateSystem)
    // TODO: const GetValues?
    .def("get_values", [](FieldDataView<Real, HostMemorySpace>& self) {
      return view_to_numpy(self.GetValues());
    });
  
  // Bind FieldT for common types
  // bind_field_t<Real>(m, "Real");
  // bind_field_t<float>(m, "Float");
  
  // Only bind double separately if Real is not already double
  bind_field_t<double>(m, "Double");
}

} // namespace pcms
