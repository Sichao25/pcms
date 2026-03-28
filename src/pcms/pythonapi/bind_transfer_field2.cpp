#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pcms/transfer/copy.h"
#include "pcms/field/field.h"
#include "pcms/field/function_space.h"
#include "pcms/field/point_evaluator.h"
#include "../transfer/interpolator.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "numpy_array_transform.h"
#include "pcms/utility/types.h"

namespace py = pybind11;

namespace pcms
{

void bind_transfer_field2_module(py::module& m)
{
  py::class_<PointEvaluator<Real>, std::unique_ptr<PointEvaluator<Real>>>(
    m, "PointEvaluator")
    .def(
      "evaluate",
      [](const PointEvaluator<Real>& self, const Field<Real>& field,
         py::array_t<Real> output) {
        auto output_view = numpy_to_view_2d<Real>(output);
        self.Evaluate(field, output_view);
      },
      py::arg("field"), py::arg("output"),
      "Evaluate the given field at the cached query coordinates into a "
      "preallocated 2D numpy array of shape "
      "(num_query_points, num_components).");

  // Bind Interpolator<Real>: construct once per source×target function-space
  // pair (localization happens at construction), then call apply() repeatedly
  // for different field states at zero additional localization cost.
  py::class_<Interpolator<Real>>(m, "Interpolator")
    .def(py::init([](const FunctionSpace& source_space,
                     const FunctionSpace& target_space,
                     OutOfBoundsPolicy policy) {
           return Interpolator<Real>(source_space, target_space, policy);
         }),
         py::arg("source_space"), py::arg("target_space"),
         py::arg("policy") = OutOfBoundsPolicy{},
         "Construct an interpolator. Localization is performed here and cached. "
         "Call apply() repeatedly without re-localizing.")
    .def(
      "apply",
      [](const Interpolator<Real>& self, const Field<Real>& source,
         Field<Real>& target) { self.Apply(source, target); },
      py::arg("source"), py::arg("target"),
      "Interpolate source field to target DOF locations (cheap; reuses cached localization).");

  py::class_<Copy<Real>>(m, "Copy")
    .def(py::init([](const FunctionSpace& source_space,
                     const FunctionSpace& target_space) {
           return Copy<Real>(source_space, target_space);
         }),
         py::arg("source_space"), py::arg("target_space"),
         "Construct a copy operator for compatible function spaces.")
    .def(
      "apply",
      [](const Copy<Real>& self, const Field<Real>& source,
         Field<Real>& target) { self.Apply(source, target); },
      py::arg("source"), py::arg("target"),
      "Copy source field data to target field (same layout required).");
}

} // namespace pcms
