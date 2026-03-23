#ifndef PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H
#define PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H

#include "point_cloud_layout.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/utility/assert.h"

#include <memory>

namespace pcms
{

// PointCloudEvaluatorFactory implements FieldEvaluatorFactory<Real> for
// point cloud (nodal) fields backed by PointCloudLayout.
class PointCloudEvaluatorFactory : public FieldEvaluatorFactory<Real>
{
public:
  explicit PointCloudEvaluatorFactory(
    std::shared_ptr<const PointCloudLayout> layout)
    : layout_(std::move(layout))
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
    CoordinateView<HostMemorySpace> /* coords */,
    OutOfBoundsPolicy /* policy */ = {}) const override
  {
    throw pcms_error(
      "PointCloudEvaluatorFactory::CreatePointEvaluator: point-cloud "
      "evaluation is not implemented");
  }

private:
  std::shared_ptr<const PointCloudLayout> layout_;
};

} // namespace pcms

#endif // PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H
