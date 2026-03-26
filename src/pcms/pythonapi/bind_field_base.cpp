#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pcms/field/coordinate_system.h"
#include "pcms/field/coordinate.h"
#include "pcms/field/field.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/function_space.h"
#include "pcms/field/data/simple.h"
#include "pcms/field/layout/uniform_grid.h"
#include "pcms/field/uniform_grid_binary_field.h"
#include "pcms/field/function_space/lagrange.h"
#include "pcms/utility/uniform_grid.h"
#include "numpy_array_transform.h"

namespace py = pybind11;

namespace pcms
{

void bind_coordinate_system_module(py::module& m)
{
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
         py::arg("coordinate_system"), py::arg("coordinates"),
         "Constructor for CoordinateView")

    .def("get_coordinate_system",
         &CoordinateView<HostMemorySpace>::GetCoordinateSystem,
         "Get the coordinate system")

    .def("set_coordinate_system",
         &CoordinateView<HostMemorySpace>::SetCoordinateSystem, py::arg("cs"),
         "Set the coordinate system")

    .def(
      "get_coordinates",
      [](const CoordinateView<HostMemorySpace>& self) {
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
  py::class_<CoordinateTransformation,
             std::shared_ptr<CoordinateTransformation>>(
    m, "CoordinateTransformation")
    .def("evaluate", &CoordinateTransformation::Evaluate, py::arg("from"),
         py::arg("to"), "Evaluate the coordinate transformation");
}

void bind_coordinate_module(py::module& m)
{
  // Bind Coordinate template for common cases
  // 3D Cartesian coordinates with Real type
  py::class_<Coordinate<CoordinateSystem, Real, 3>>(m, "Coordinate3D")
    .def(py::init<Real, Real, Real>(), py::arg("x"), py::arg("y"), py::arg("z"),
         "Constructor for 3D coordinate")

    .def(
      "values",
      [](const Coordinate<CoordinateSystem, Real, 3>& self) {
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
         py::arg("index"), "Get coordinate value by index");

  // 2D coordinates
  py::class_<Coordinate<CoordinateSystem, Real, 2>>(m, "Coordinate2D")
    .def(py::init<Real, Real>(), py::arg("x"), py::arg("y"),
         "Constructor for 2D coordinate")

    .def(
      "values",
      [](const Coordinate<CoordinateSystem, Real, 2>& self) {
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
         py::arg("index"), "Get coordinate value by index");

  // Bind CoordinateElement for common types
  py::class_<CoordinateElement<CoordinateSystem, Real>>(m, "CoordinateElement")
    .def(py::init<Real>(), py::arg("data"), "Constructor for CoordinateElement")

    .def("underlying",
         py::overload_cast<>(
           &CoordinateElement<CoordinateSystem, Real>::underlying, py::const_),
         "Get the underlying value");
}

void bind_create_field_module(py::module& m)
{
  // Bind FieldLayout abstract base with shared_ptr holder so Python can hold
  // references returned by LagrangeFunctionSpace::get_layout().
  py::class_<FieldLayout, std::shared_ptr<FieldLayout>>(m, "FieldLayout")
    .def("get_num_components", &FieldLayout::GetNumComponents,
         "Number of field components per DOF holder");

  // Bind Field<Real>: composed per-field object returned by
  // FunctionSpace-backed factories' create_field(). Move-only in C++; Python
  // holds it by value in a heap-allocated wrapper.
  py::class_<Field<Real>>(m, "Field")
    .def(
      "get_dof_holder_data",
      [](const Field<Real>& self) {
        auto data = self.GetDOFHolderDataHost();
        py::array_t<Real> result(static_cast<py::ssize_t>(data.size()));
        auto buf = result.request();
        Real* ptr = static_cast<Real*>(buf.ptr);
        for (size_t i = 0; i < data.size(); ++i)
          ptr[i] = data[i];
        return result;
      },
      "Get the DOF holder data as a 1D numpy array")

    .def(
      "set_dof_holder_data",
      [](Field<Real>& self, py::array_t<const Real> arr) {
        auto buf = arr.request();
        if (buf.ndim != 1) {
          throw std::runtime_error("DOF holder data must be a 1D array");
        }
        Rank1View<const Real, HostMemorySpace> view(
          static_cast<const Real*>(buf.ptr),
          static_cast<size_t>(buf.shape[0]));
        self.SetDOFHolderDataHost(view);
      },
      py::arg("data"), "Set the DOF holder data from a 1D numpy array")

    .def(
      "get_num_dof_holders",
      [](const Field<Real>& self) {
        return self.GetLayout().GetNumOwnedDofHolder();
      },
      "Number of owned DOF holders (nodes/elements)")

    .def(
      "get_num_components",
      [](const Field<Real>& self) {
        return self.GetLayout().GetNumComponents();
      },
      "Number of field components per DOF holder")

    .def(
      "get_dof_holder_coordinates",
      [](const Field<Real>& self) {
        auto cv    = self.GetData().GetDOFHolderCoordinatesHost();
        auto coords = cv.GetCoordinates();
        py::array_t<Real> result({static_cast<py::ssize_t>(coords.extent(0)),
                                   static_cast<py::ssize_t>(coords.extent(1))});
        auto buf = result.request();
        Real* ptr = static_cast<Real*>(buf.ptr);
        for (size_t i = 0; i < coords.extent(0); ++i)
          for (size_t j = 0; j < coords.extent(1); ++j)
            ptr[i * coords.extent(1) + j] = coords(i, j);
        return result;
      },
      "DOF holder coordinates as a 2D numpy array (num_dof_holders × dim)");

  py::class_<FunctionSpace>(m, "FunctionSpace")
    .def(
      "get_layout",
      [](const FunctionSpace& self) -> std::shared_ptr<FieldLayout> {
        return std::const_pointer_cast<FieldLayout>(self.GetLayout());
      },
      "Get the field layout")

    .def(
      "get_evaluator_factory",
      [](const FunctionSpace& self) -> const FieldEvaluatorFactory<Real>& {
        return self.GetEvaluatorFactory();
      },
      py::return_value_policy::reference_internal,
      "Get the FieldEvaluatorFactory for this function space")

    .def("get_coordinate_system", &FunctionSpace::GetCoordinateSystem,
         "Get the coordinate system for this function space");

  // Bind LagrangeFunctionSpace
  py::class_<LagrangeFunctionSpace, FunctionSpace>(m, "LagrangeFunctionSpace")
    .def_static(
      "from_mesh",
      [](Omega_h::Mesh& mesh, int order, int num_components,
         CoordinateSystem coordinate_system) {
        return LagrangeFunctionSpace::FromMesh(mesh, order, num_components,
                                             coordinate_system);
      },
      py::arg("mesh"), py::arg("order"), py::arg("num_components") = 1,
      py::arg("coordinate_system") = CoordinateSystem::Cartesian,
      "Create a LagrangeFunctionSpace from an Omega_h mesh")

    .def_static(
      "from_uniform_grid",
      [](const UniformGrid<2>& grid, int num_components, CoordinateSystem cs,
         int order) {
        return LagrangeFunctionSpace::FromUniformGrid(grid, num_components, cs,
                                                      order);
      },
      py::arg("grid"), py::arg("num_components") = 1,
      py::arg("coordinate_system") = CoordinateSystem::Cartesian,
      py::arg("order") = 1,
      "Create a LagrangeFunctionSpace from a 2D uniform grid")

    .def_static(
      "from_uniform_grid",
      [](const UniformGrid<3>& grid, int num_components, CoordinateSystem cs,
         int order) {
        return LagrangeFunctionSpace::FromUniformGrid(grid, num_components, cs,
                                                      order);
      },
      py::arg("grid"), py::arg("num_components") = 1,
      py::arg("coordinate_system") = CoordinateSystem::Cartesian,
      py::arg("order") = 1,
      "Create a LagrangeFunctionSpace from a 3D uniform grid")

    .def(
      "create_field",
      [](const LagrangeFunctionSpace& self) { return self.CreateField(); },
      "Create a Field<Real> for this function space");

  // Bind CreateUniformGridFromMesh for 2D
  m.def(
    "create_uniform_grid_from_mesh",
    [](Omega_h::Mesh& mesh, const std::array<LO, 2>& divisions) {
      return CreateUniformGridFromMesh<2>(mesh, divisions);
    },
    py::arg("mesh"), py::arg("divisions"),
    "Create a 2D uniform grid from an Omega_h mesh");

  // Bind CreateUniformGridFromMesh for 3D
  m.def(
    "create_uniform_grid_from_mesh",
    [](Omega_h::Mesh& mesh, const std::array<LO, 3>& divisions) {
      return CreateUniformGridFromMesh<3>(mesh, divisions);
    },
    py::arg("mesh"), py::arg("divisions"),
    "Create a 3D uniform grid from an Omega_h mesh");

  m.def(
    "create_uniform_grid_binary_field",
    [](Omega_h::Mesh& mesh, const std::array<LO, 2>& divisions) -> py::tuple {
      auto [layout, field] =
        CreateUniformGridBinaryField<2>(mesh, divisions);
      auto mask_field = Field<Real>(nullptr, std::move(field));
      return py::make_tuple(layout, std::move(mask_field));
    },
    py::arg("mesh"), py::arg("divisions"),
    "Create a 2D vertex mask field indicating inside/outside mesh");
}

// bind_field_module is kept for compatibility but now registers nothing that
// refers to the deleted FieldT / LocalizationHint / FieldDataView types.
void bind_field_module(py::module& /*m*/) {}

} // namespace pcms
