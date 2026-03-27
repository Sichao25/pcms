#ifndef PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H
#define PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H

#include "pcms/field/layout/point_cloud.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/field/evaluator/mls_point_cloud.h"
#include "pcms/localization/mls_support_helpers.h"
#include "pcms/utility/assert.h"

#include <Omega_h_array.hpp>
#include <memory>

namespace pcms
{

// PointCloudEvaluatorFactory implements FieldEvaluatorFactory<Real> for
// point-cloud (nodal) fields backed by PointCloudLayout.
//
// Point-cloud evaluation is implemented using Moving Least Squares (MLS).
// CreatePointEvaluator performs support localization once for the supplied
// target coordinates and returns an MLSPointEvaluator that can be reused
// across repeated Evaluate calls at zero additional localization cost.
class PointCloudEvaluatorFactory : public FieldEvaluatorFactory<Real>
{
public:
  explicit PointCloudEvaluatorFactory(
    std::shared_ptr<const PointCloudLayout> layout, MLSOptions options = {})
    : layout_(std::move(layout)), options_(options)
  {
  }

  const FieldLayout& GetLayout() const override { return *layout_; }

  CoordinateSystem GetCoordinateSystem() const override
  {
    return layout_->GetDOFHolderCoordinates().GetCoordinateSystem();
  }

  bool HasDOFHolderCoordinates() const override { return true; }

  CoordinateView<HostMemorySpace> GetDOFHolderCoordinatesHost() const override
  {
    return layout_->GetDOFHolderCoordinates();
  }

  bool SupportsNearestBoundary() const override { return false; }

  std::unique_ptr<PointEvaluator<Real>> CreatePointEvaluator(
    CoordinateView<HostMemorySpace> target_coords,
    OutOfBoundsPolicy /* policy */ = {}) const override
  {
    if (target_coords.GetCoordinateSystem() != GetCoordinateSystem()) {
      throw pcms_error(
        "PointCloudEvaluatorFactory: coordinate system mismatch");
    }
    if (GetCoordinateSystem() != CoordinateSystem::Cartesian) {
      throw pcms_error(
        "PointCloudEvaluatorFactory: only Cartesian coordinates are "
        "supported for MLS point-cloud evaluation");
    }

    const auto src_view = layout_->GetCoordinatesHost();
    const int dim = layout_->GetDimension();
    const int n_src = static_cast<int>(src_view.extent(0));

    // Convert source rank-2 host view → flat Omega_h::Reals
    Omega_h::HostWrite<Omega_h::Real> src_hw(n_src * dim, "src_coords");
    for (int i = 0; i < n_src; ++i)
      for (int d = 0; d < dim; ++d)
        src_hw[i * dim + d] = src_view(i, d);
    Omega_h::Reals source_coords(src_hw);

    // Convert target rank-2 host view → flat Omega_h::Reals
    const auto tgt_view = target_coords.GetCoordinates(); // [n_tgt][dim]
    PCMS_ALWAYS_ASSERT(static_cast<int>(tgt_view.extent(1)) == dim);
    const int n_tgt = static_cast<int>(tgt_view.extent(0));
    Omega_h::HostWrite<Omega_h::Real> tgt_hw(n_tgt * dim, "tgt_coords");
    for (int i = 0; i < n_tgt; ++i)
      for (int d = 0; d < dim; ++d)
        tgt_hw[i * dim + d] = tgt_view(i, d);
    Omega_h::Reals target_coords_oh(tgt_hw);

    auto supports = BuildPointCloudSupports(
      source_coords, target_coords_oh, dim, options_.radius,
      options_.min_req_supports, options_.adapt_radius);

    return std::make_unique<MLSPointEvaluator>(
      std::move(source_coords), std::move(target_coords_oh),
      std::move(supports), dim, options_);
  }

private:
  std::shared_ptr<const PointCloudLayout> layout_;
  MLSOptions options_;
};

} // namespace pcms

#endif // PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H
