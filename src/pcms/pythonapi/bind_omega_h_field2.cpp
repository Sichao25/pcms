#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "pcms/adapter/omega_h/omega_h_field2.h"
#include "pcms/adapter/omega_h/omega_h_field_layout.h"
#include "helpers.h"

namespace py = pybind11;

namespace pcms {

void bind_omega_h_field2(py::module& m) {
  // Bind LocalizationHint
  py::class_<LocalizationHint, std::shared_ptr<LocalizationHint>>(m, "LocalizationHint")
    .def(py::init<>(), "Default constructor");
  
  // Bind MeshFieldBackend interface
  py::class_<MeshFieldBackend, std::shared_ptr<MeshFieldBackend>>(m, "MeshFieldBackend")
    .def("evaluate", [](const MeshFieldBackend& self,
                       py::array_t<Real> local_coords,
                       py::array_t<LO> offsets) {
      auto coords_view = numpy_to_kokkos_view_2d<Real>(local_coords);
      auto offsets_view = numpy_to_kokkos_view<LO>(offsets);
      auto coords_view_device = Kokkos::View<Real**>("coords_view_device", 
                                                        coords_view.extent(0), 
                                                        coords_view.extent(1));
      Kokkos::deep_copy(coords_view_device, coords_view);
      auto offsets_view_device = Kokkos::View<LO*>("offsets_view_device", 
                                                        offsets_view.extent(0));
      Kokkos::deep_copy(offsets_view_device, offsets_view);
      auto result = self.evaluate(coords_view_device, offsets_view_device);
      
      // Convert result to numpy array
      Kokkos::View<Real**, HostMemorySpace> result_h("result_h", 
                                                      result.extent(0), 
                                                      result.extent(1));
      Kokkos::deep_copy(result_h, result);
      return kokkos_view_2d_to_numpy(result_h);
    },
    py::arg("local_coords"),
    py::arg("offsets"),
    "Evaluate the field at local coordinates")
    
    .def("set_data", [](MeshFieldBackend& self,
                       py::array_t<const Real> data,
                       size_t num_nodes,
                       size_t num_components,
                       int dim) {
      auto data_view = numpy_to_view<const Real>(data);
      self.SetData(data_view, num_nodes, num_components, dim);
    },
    py::arg("data"),
    py::arg("num_nodes"),
    py::arg("num_components"),
    py::arg("dim"),
    "Set the field data")
    
    .def("get_data", [](const MeshFieldBackend& self,
                       size_t total_size,
                       size_t num_nodes,
                       size_t num_components,
                       int dim) {
      Kokkos::View<Real*, HostMemorySpace> data("data", total_size);
      Rank1View<Real, HostMemorySpace> data_view(data.data(), data.extent(0));
      self.GetData(data_view, num_nodes, num_components, dim);
      return kokkos_view_to_numpy(data);
    },
    py::arg("total_size"),
    py::arg("num_nodes"),
    py::arg("num_components"),
    py::arg("dim"),
    "Get the field data");

  // Bind OmegaHField2 class
  py::class_<OmegaHField2, FieldT<Real>, std::shared_ptr<OmegaHField2>>(m, "OmegaHField2")
    .def(py::init<const OmegaHFieldLayout&>(),
         py::arg("layout"),
         "Constructor for OmegaHField2")
    
    .def("get_localization_hint", [](const OmegaHField2& self,
                                     py::array_t<Real> coordinates,
                                     const CoordinateSystem& coord_system) {
      // Create CoordinateView from numpy array
      auto coords_view = numpy_to_view_2d<const Real>(coordinates);
      CoordinateView<HostMemorySpace> coord_view(coord_system, coords_view);
      return self.GetLocalizationHint(coord_view);
    },
    py::arg("coordinates"),
    py::arg("coordinate_system"),
    "Get localization hint for given coordinates")
    
    .def("evaluate", [](const OmegaHField2& self,
                       const LocalizationHint& location,
                       py::array_t<Real> values,
                       const CoordinateSystem& coord_system) {
      // Create FieldDataView
      auto values_view = numpy_to_view<Real>(values);
      FieldDataView<Real, HostMemorySpace> field_data_view(values_view, coord_system);
      self.Evaluate(location, field_data_view);
    },
    py::arg("location"),
    py::arg("values"),
    py::arg("coordinate_system"),
    "Evaluate field at locations specified by localization hint")
    
    .def("evaluate_gradient", [](OmegaHField2& self,
                                py::array_t<Real> gradients,
                                const CoordinateSystem& coord_system) {
      auto gradients_view = numpy_to_view<Real>(gradients);
      FieldDataView<Real, HostMemorySpace> field_data_view(gradients_view, coord_system);
      self.EvaluateGradient(field_data_view);
    },
    py::arg("gradients"),
    py::arg("coordinate_system"),
    "Evaluate gradient of the field")
    
    .def("get_layout", &OmegaHField2::GetLayout,
         py::return_value_policy::reference,
         "Get the field layout")
    
    .def("can_evaluate_gradient", &OmegaHField2::CanEvaluateGradient,
         "Check if gradient evaluation is supported")
    
    .def("serialize", [](const OmegaHField2& self,
                        py::array_t<Real> buffer,
                        py::array_t<const pcms::LO> permutation) {
      auto buffer_view = numpy_to_view<Real>(buffer);
      auto perm_view = numpy_to_view<const pcms::LO>(permutation);
      return self.Serialize(buffer_view, perm_view);
    },
    py::arg("buffer"),
    py::arg("permutation"),
    "Serialize field data into buffer")
    
    .def("deserialize", [](OmegaHField2& self,
                          py::array_t<const Real> buffer,
                          py::array_t<const pcms::LO> permutation) {
      auto buffer_view = numpy_to_view<const Real>(buffer);
      auto perm_view = numpy_to_view<const pcms::LO>(permutation);
      self.Deserialize(buffer_view, perm_view);
    },
    py::arg("buffer"),
    py::arg("permutation"),
    "Deserialize field data from buffer")
    
    .def("get_dof_holder_data", [](const OmegaHField2& self) {
      auto data = self.GetDOFHolderData();
      // Create a copy since we're returning to Python
      py::array_t<Real> result = view_to_numpy(data);
      return result;
    },
    "Get the DOF holder data")
    
    .def("set_dof_holder_data", [](OmegaHField2& self,
                                   py::array_t<const Real> data) {
      auto data_view = numpy_to_view<const Real>(data);
      self.SetDOFHolderData(data_view);
    },
    py::arg("data"),
    "Set the DOF holder data");

  // Helper functions for creating views (if needed for testing)
  m.def("create_coordinate_view", [](py::array_t<Real> coordinates,
                                     const CoordinateSystem& coord_system) {
    auto coords_view = numpy_to_view_2d<const Real>(coordinates);
    return CoordinateView<HostMemorySpace>(coord_system, coords_view);
  },
  py::arg("coordinates"),
  py::arg("coordinate_system"),
  "Create a CoordinateView from numpy array");

  m.def("create_field_data_view", [](py::array_t<Real> values,
                                     const CoordinateSystem& coord_system) {
    auto values_view = numpy_to_view<Real>(values);
    return FieldDataView<Real, HostMemorySpace>(values_view, coord_system);
  },
  py::arg("values"),
  py::arg("coordinate_system"),
  "Create a FieldDataView from numpy array");
}

} // namespace pcms
