#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pcms/coordinate_system.h"
#include "pcms/coordinate.h"
#include "pcms/create_field.h"
#include "helpers.h"

namespace py = pybind11;

namespace pcms {

void bind_coordinate_system_module(py::module& m) {
  // Bind CoordinateSystem enum
  py::enum_<CoordinateSystem>(m, "CoordinateSystem")
    .value("Cartesian", CoordinateSystem::Cartesian)
    .value("Cylindrical", CoordinateSystem::Cylindrical)
    .value("XGC", CoordinateSystem::XGC)
    .value("GNET", CoordinateSystem::GNET)
    .value("BEAMS3D", CoordinateSystem::BEAMS3D)
    .export_values();
  
  // Bind CoordinateView for HostMemorySpace
  py::class_<CoordinateView<HostMemorySpace>>(m, "CoordinateView")
    .def(py::init([](CoordinateSystem cs, py::array_t<Real> coords) {
      // Convert 2D numpy array to Rank2View
      auto buf = coords.request();
      if (buf.ndim != 2) {
        throw std::runtime_error("Coordinates must be a 2D array");
      }
      // Create a view from the numpy array
      Rank2View<const Real, HostMemorySpace> coords_view(
        static_cast<Real*>(buf.ptr), buf.shape[0], buf.shape[1]);
      return CoordinateView<HostMemorySpace>(cs, coords_view);
    }),
    py::arg("coordinate_system"),
    py::arg("coordinates"),
    "Constructor for CoordinateView")
    
    .def("get_coordinate_system", &CoordinateView<HostMemorySpace>::GetCoordinateSystem,
         "Get the coordinate system")
    
    .def("set_coordinate_system", &CoordinateView<HostMemorySpace>::SetCoordinateSystem,
         py::arg("cs"),
         "Set the coordinate system")
    
    .def("get_coordinates", [](const CoordinateView<HostMemorySpace>& self) {
      auto coords = self.GetCoordinates();
      // Convert to numpy array
      py::array_t<Real> result({static_cast<py::ssize_t>(coords.extent(0)),
                                 static_cast<py::ssize_t>(coords.extent(1))});
      auto buf = result.request();
      Real* ptr = static_cast<Real*>(buf.ptr);
      for (size_t i = 0; i < coords.extent(0); ++i) {
        for (size_t j = 0; j < coords.extent(1); ++j) {
          ptr[i * coords.extent(1) + j] = coords(i, j);
        }
      }
      return result;
    },
    "Get the coordinates as numpy array");
  
  // Bind CoordinateTransformation abstract base class
  py::class_<CoordinateTransformation, std::shared_ptr<CoordinateTransformation>>(
    m, "CoordinateTransformation")
    .def("evaluate", &CoordinateTransformation::Evaluate,
         py::arg("from"),
         py::arg("to"),
         "Evaluate the coordinate transformation");
}

void bind_coordinate_module(py::module& m) {
  // Bind Coordinate template for common cases
  // 3D Cartesian coordinates with Real type
  py::class_<Coordinate<CoordinateSystem, Real, 3>>(m, "Coordinate3D")
    .def(py::init<Real, Real, Real>(),
         py::arg("x"),
         py::arg("y"),
         py::arg("z"),
         "Constructor for 3D coordinate")
    
    .def("values", [](const Coordinate<CoordinateSystem, Real, 3>& self) {
      auto vals = self.Values();
      py::array_t<Real> result(3);
      auto buf = result.request();
      Real* ptr = static_cast<Real*>(buf.ptr);
      ptr[0] = vals[0];
      ptr[1] = vals[1];
      ptr[2] = vals[2];
      return result;
    },
    "Get coordinate values as numpy array")
    
    .def("__getitem__", &Coordinate<CoordinateSystem, Real, 3>::operator[],
         py::arg("index"),
         "Get coordinate value by index");
  
  // 2D coordinates
  py::class_<Coordinate<CoordinateSystem, Real, 2>>(m, "Coordinate2D")
    .def(py::init<Real, Real>(),
         py::arg("x"),
         py::arg("y"),
         "Constructor for 2D coordinate")
    
    .def("values", [](const Coordinate<CoordinateSystem, Real, 2>& self) {
      auto vals = self.Values();
      py::array_t<Real> result(2);
      auto buf = result.request();
      Real* ptr = static_cast<Real*>(buf.ptr);
      ptr[0] = vals[0];
      ptr[1] = vals[1];
      return result;
    },
    "Get coordinate values as numpy array")
    
    .def("__getitem__", &Coordinate<CoordinateSystem, Real, 2>::operator[],
         py::arg("index"),
         "Get coordinate value by index");
  
  // Bind CoordinateElement for common types
  py::class_<CoordinateElement<CoordinateSystem, Real>>(m, "CoordinateElement")
    .def(py::init<Real>(),
         py::arg("data"),
         "Constructor for CoordinateElement")
    
    .def("underlying", 
         py::overload_cast<>(&CoordinateElement<CoordinateSystem, Real>::underlying, py::const_),
         "Get the underlying value");
}

void bind_create_field_module(py::module& m) {
  // Bind CreateLagrangeLayout function
  m.def("create_lagrange_layout",
        [](Omega_h::Mesh& mesh, 
           int order, 
           int num_components,
           CoordinateSystem coordinate_system) {
          return CreateLagrangeLayout(mesh, order, num_components, coordinate_system);
        },
        py::arg("mesh"),
        py::arg("order"),
        py::arg("num_components"),
        py::arg("coordinate_system"),
        R"doc(
        Create a Lagrange field layout
        
        Parameters
        ----------
        mesh : OmegaHMesh
            The mesh to create the layout on
        order : int
            The polynomial order of the Lagrange elements
        num_components : int
            Number of components per DOF
        coordinate_system : CoordinateSystem
            The coordinate system for the field
        
        Returns
        -------
        FieldLayout
            A unique pointer to the created field layout
        )doc");
}

} // namespace pcms
