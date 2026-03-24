#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pcms/utility/arrays.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/field/coordinate.h"
#include "pcms/utility/memory_spaces.h"
#include "numpy_array_transform.h"

namespace py = pybind11;

namespace pcms
{

void bind_omega_h_field2(py::module& m)
{
  // Bind OutOfBoundsMode enum
  py::enum_<OutOfBoundsMode>(m, "OutOfBoundsMode")
    .value("ERROR", OutOfBoundsMode::ERROR,
           "Raise error when points are out of bounds")
    .value("FILL", OutOfBoundsMode::FILL,
           "Fill with a specified value when points are out of bounds")
    .value("NEAREST_BOUNDARY", OutOfBoundsMode::NEAREST_BOUNDARY,
           "Clamp to nearest boundary cell (extrapolate)")
    .export_values();

  // Bind OutOfBoundsPolicy struct
  py::class_<OutOfBoundsPolicy>(m, "OutOfBoundsPolicy")
    .def(py::init<>(), "Default constructor (mode=ERROR, fill_value=0)")
    .def(py::init([](OutOfBoundsMode mode, Real fill_value) {
           OutOfBoundsPolicy p;
           p.mode       = mode;
           p.fill_value = fill_value;
           return p;
         }),
         py::arg("mode"), py::arg("fill_value") = Real(0),
         "Construct with explicit mode and optional fill value")
    .def_readwrite("mode", &OutOfBoundsPolicy::mode, "Out-of-bounds handling mode")
    .def_readwrite("fill_value", &OutOfBoundsPolicy::fill_value,
                   "Fill value (used only when mode == FILL)");

  // Helper: create a CoordinateView from a 2D numpy array
  m.def(
    "create_coordinate_view",
    [](py::array_t<Real> coordinates, const CoordinateSystem& coord_system) {
      auto coords_view = numpy_to_view_2d<const Real>(coordinates);
      return CoordinateView<HostMemorySpace>(coord_system, coords_view);
    },
    py::arg("coordinates"), py::arg("coordinate_system"),
    "Create a CoordinateView from a 2D numpy array");

  // NOTE: MeshFieldsAdapter2<Real> and FieldT<Real> have been removed from
  // the C++ API. Fields are now represented as FieldData<Real> (bound in
  // bind_field_base.cpp) and created via LagrangeFunctionSpace::create_field_data().
}

} // namespace pcms
