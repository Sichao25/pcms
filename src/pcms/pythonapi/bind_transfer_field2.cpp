#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pcms/transfer/copy.h"
#include "pcms/field/field.h"
#include "pcms/field/field_data.h"
#include "../transfer/interpolator.h"
#include "pcms/field/lagrange_field_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/utility/types.h"

namespace py = pybind11;

namespace pcms
{

void bind_transfer_field2_module(py::module& m)
{
  // Bind Interpolator<Real>: construct once per source×target function-space
  // pair (localization happens at construction), then call apply() repeatedly
  // for different field states at zero additional localization cost.
  py::class_<Interpolator<Real>>(m, "Interpolator")
    .def(py::init([](const LagrangeFunctionSpace& source_space,
                     const LagrangeFunctionSpace& target_space,
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

  // copy_field: source and target must share the same layout.
  m.def(
    "copy_field",
    [](const FieldData<Real>& source, FieldData<Real>& target) {
      copy_field2(source, target);
    },
    py::arg("source"), py::arg("target"),
    "Copy field data from source to target (same layout required).");
}

} // namespace pcms
