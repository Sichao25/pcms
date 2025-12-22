#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pcms/interpolator/interpolation_base.h"
#include "helpers.h"

namespace py = pybind11;

namespace pcms {

void bind_interpolation_base_module(py::module& m) {
  // Bind InterpolationBase abstract base class
  py::class_<InterpolationBase, std::shared_ptr<InterpolationBase>>(m, "InterpolationBase")
    .def("eval", [](InterpolationBase& self,
                   py::array_t<double> source_field,
                   py::array_t<double> target_field) {
      auto source_view = numpy_to_view<double>(source_field);
      auto target_view = numpy_to_view<double>(target_field);
      self.eval(source_view, target_view);
    },
    py::arg("source_field"),
    py::arg("target_field"),
    "Evaluate the interpolation from source to target field")
    
    .def("get_source_size", &InterpolationBase::getSourceSize,
         "Get the size of the source field")
    
    .def("get_target_size", &InterpolationBase::getTargetSize,
         "Get the size of the target field");
  
  // Bind MLSPointCloudInterpolation
  py::class_<MLSPointCloudInterpolation, InterpolationBase, 
             std::shared_ptr<MLSPointCloudInterpolation>>(m, "MLSPointCloudInterpolation")
    .def(py::init([](py::array_t<double> source_points,
                     py::array_t<double> target_points,
                     int dim,
                     double radius,
                     unsigned min_req_supports,
                     unsigned degree,
                     bool adapt_radius,
                     double lambda,
                     double decay_factor) {
      auto source_view = numpy_to_view<double>(source_points);
      auto target_view = numpy_to_view<double>(target_points);
      return new MLSPointCloudInterpolation(
        source_view, target_view, dim, radius, 
        min_req_supports, degree, adapt_radius, lambda, decay_factor);
    }),
    py::arg("source_points"),
    py::arg("target_points"),
    py::arg("dim"),
    py::arg("radius"),
    py::arg("min_req_supports") = 10,
    py::arg("degree") = 3,
    py::arg("adapt_radius") = true,
    py::arg("lambda") = 0.0,
    py::arg("decay_factor") = 5.0,
    R"doc(
    Constructor for point-cloud based MLS interpolation
    
    Parameters
    ----------
    source_points : numpy array
        Source point coordinates (flattened: [x1, y1, z1, x2, y2, z2, ...])
    target_points : numpy array
        Target point coordinates (flattened)
    dim : int
        Spatial dimension of the points
    radius : float
        Cutoff radius for the MLS interpolation
    min_req_supports : int, optional
        Minimum number of source locations required (default: 10)
    degree : int, optional
        Degree of the polynomial used in MLS (default: 3)
    adapt_radius : bool, optional
        Whether to adapt radius to satisfy min/max supports (default: True)
    lambda : float, optional
        Regularization parameter (default: 0.0)
    decay_factor : float, optional
        Decay factor for the weight function (default: 5.0)
    )doc")
    
    .def("get_supports", &MLSPointCloudInterpolation::getSupports,
         "Get the support structure for MLS interpolation");
  
  // Bind MLSMeshInterpolation
  py::class_<MLSMeshInterpolation, InterpolationBase,
             std::shared_ptr<MLSMeshInterpolation>>(m, "MLSMeshInterpolation")
    // Vertex to Vertex interpolation between two meshes
    .def(py::init<Omega_h::Mesh&, Omega_h::Mesh&, double, unsigned, unsigned, 
                  bool, double, double>(),
         py::arg("source_mesh"),
         py::arg("target_mesh"),
         py::arg("radius"),
         py::arg("min_req_supports") = 10,
         py::arg("degree") = 3,
         py::arg("adapt_radius") = true,
         py::arg("lambda") = 0.0,
         py::arg("decay_factor") = 5.0,
         R"doc(
         Vertex to Vertex interpolation between two meshes
         
         Parameters
         ----------
         source_mesh : OmegaHMesh
             Source mesh
         target_mesh : OmegaHMesh
             Target mesh
         radius : float
             Cutoff radius for the MLS interpolation
         min_req_supports : int, optional
             Minimum number of source locations required (default: 10)
         degree : int, optional
             Degree of the polynomial (default: 3)
         adapt_radius : bool, optional
             Whether to adapt radius (default: True)
         lambda : float, optional
             Regularization parameter (default: 0.0)
         decay_factor : float, optional
             Decay factor for weight function (default: 5.0)
         )doc")
    
    // Centroid to Vertex interpolation within a single mesh
    .def(py::init<Omega_h::Mesh&, double, unsigned, unsigned, bool, double, double>(),
         py::arg("source_mesh"),
         py::arg("radius"),
         py::arg("min_req_supports") = 10,
         py::arg("degree") = 3,
         py::arg("adapt_radius") = true,
         py::arg("lambda") = 0.0,
         py::arg("decay_factor") = 5.0,
         R"doc(
         Centroid to Vertex interpolation within a single mesh
         
         Parameters
         ----------
         source_mesh : OmegaHMesh
             The mesh to interpolate within
         radius : float
             Cutoff radius for the MLS interpolation
         min_req_supports : int, optional
             Minimum number of source locations required (default: 10)
         degree : int, optional
             Degree of the polynomial (default: 3)
         adapt_radius : bool, optional
             Whether to adapt radius (default: True)
         lambda : float, optional
             Regularization parameter (default: 0.0)
         decay_factor : float, optional
             Decay factor for weight function (default: 5.0)
         )doc")
    
    .def("get_supports", &MLSMeshInterpolation::getSupports,
         "Get the support structure for MLS interpolation");
}

} // namespace pcms
