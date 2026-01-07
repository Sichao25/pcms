#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "pcms/adapter/xgc/xgc_field_adapter.h"
#include "helpers.h"

namespace py = pybind11;

namespace pcms {

template<typename T, typename CoordinateElementType>
void bind_xgc_field_adapter(py::module& m, const std::string& type_suffix) {
  using XGCFieldAdapterType = XGCFieldAdapter<T, CoordinateElementType>;
  
  std::string class_name = "XGCFieldAdapter" + type_suffix;
  
  py::class_<XGCFieldAdapterType>(m, class_name.c_str())
    .def(py::init([](const std::string& name,
                     py::object plane_communicator,
                     py::array_t<T> data,
                     const ReverseClassificationVertex& reverse_classification,
                     std::function<int8_t(int, int)> in_overlap) {
      // Convert MPI communicator from mpi4py
      // Note: This requires mpi4py to be installed
      MPI_Comm comm = MPI_COMM_WORLD; // Default fallback
      
      if (!plane_communicator.is_none()) {
        // Extract MPI_Comm from mpi4py.MPI.Comm object
        // This uses the mpi4py C-API
        PyObject* mpi_comm_ptr = plane_communicator.ptr();
        if (PyObject_HasAttrString(mpi_comm_ptr, "ob_mpi")) {
          PyObject* ob_mpi = PyObject_GetAttrString(mpi_comm_ptr, "ob_mpi");
          comm = *static_cast<MPI_Comm*>(PyCapsule_GetPointer(ob_mpi, nullptr));
          Py_DECREF(ob_mpi);
        }
      }
      
      // Convert numpy array to Kokkos view
      auto kokkos_data = numpy_to_view<T>(data);
      
      return new XGCFieldAdapterType(
        name, comm, kokkos_data, reverse_classification, in_overlap);
    }))
    
    .def("serialize", [](const XGCFieldAdapterType& self,
                        py::array_t<T> buffer,
                        py::array_t<const pcms::LO> permutation) {
      auto kokkos_buffer = numpy_to_view<T>(buffer);
      auto kokkos_perm = numpy_to_view<const pcms::LO>(permutation);
      return self.Serialize(kokkos_buffer, kokkos_perm);
    })
    
    .def("deserialize", [](const XGCFieldAdapterType& self,
                          py::array_t<const T> buffer,
                          py::array_t<const pcms::LO> permutation) {
      auto kokkos_buffer = numpy_to_view<const T>(buffer);
      auto kokkos_perm = numpy_to_view<const pcms::LO>(permutation);
      self.Deserialize(kokkos_buffer, kokkos_perm);
    })
    
    .def("get_gids", &XGCFieldAdapterType::GetGids)
    
    .def("get_reverse_partition_map", &XGCFieldAdapterType::GetReversePartitionMap,
         py::arg("partition"))
    
    .def("rank_participates_coupling_communication",
         &XGCFieldAdapterType::RankParticipatesCouplingCommunication)
    
    .def("get_entity_type", &XGCFieldAdapterType::GetEntityType);
}

void bind_xgc_field_adapter_module(py::module& m) {
  // Bind ReverseClassificationVertex if not already bound
  // This is a placeholder - actual implementation depends on the definition
  
  // Bind ReadXGCNodeClassificationResult
  py::class_<ReadXGCNodeClassificationResult>(m, "ReadXGCNodeClassificationResult")
    .def(py::init<>())
    .def_readwrite("dimension", &ReadXGCNodeClassificationResult::dimension)
    .def_readwrite("geometric_id", &ReadXGCNodeClassificationResult::geometric_id);
  
  // Bind ReadXGCNodeClassification function
  m.def("read_xgc_node_classification",
        [](const std::string& input_str) {
          std::istringstream iss(input_str);
          return ReadXGCNodeClassification(iss);
        },
        py::arg("input_str"));
  
  // Bind common template instantiations
  bind_xgc_field_adapter<double, double>(m, "DD");
  bind_xgc_field_adapter<float, float>(m, "FF");
  bind_xgc_field_adapter<double, float>(m, "DF");
  bind_xgc_field_adapter<float, double>(m, "FD");
  
  // Bind evaluation functions for double/double instantiation
  m.def("get_nodal_coordinates",
        [](const XGCFieldAdapter<double, double>& field) {
          auto coords = get_nodal_coordinates(field);
          return kokkos_view_to_numpy(coords);
        },
        py::arg("field"));
  
  m.def("evaluate_lagrange",
        [](const XGCFieldAdapter<double, double>& field,
           py::array_t<double> coordinates) {
          auto kokkos_coords = numpy_to_view<const double>(coordinates);
          auto result = evaluate(field, Lagrange<1>{}, kokkos_coords);
          return kokkos_view_to_numpy(result);
        });
  
  m.def("evaluate_nearest_neighbor",
        [](const XGCFieldAdapter<double, double>& field,
           py::array_t<double> coordinates) {
          auto kokkos_coords = numpy_to_view<const double>(coordinates);
          auto result = evaluate(field, NearestNeighbor{}, kokkos_coords);
          return kokkos_view_to_numpy(result);
        });
  
  m.def("set_nodal_data",
        [](const XGCFieldAdapter<double, double>& field,
           py::array_t<double> data) {
          auto kokkos_data = numpy_to_view<const double>(data);
          set_nodal_data(field, kokkos_data);
        });
}

} // namespace pcms
