#include "pcms/field/data/point_cloud.h"
#include "pcms/field/layout/point_cloud.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/assert.h"

namespace pcms
{

PointCloud::PointCloud(std::shared_ptr<const PointCloudLayout> layout)
  : layout_(std::move(layout)),
    metadata_{},
    data_("", layout_->GetDOFHolderCoordinates().GetCoordinates().extent(0)),
    data_host_("", layout_->GetDOFHolderCoordinates().GetCoordinates().extent(0))
{
}

const FieldLayout& PointCloud::GetLayout() const
{
  return *layout_;
}

const FieldMetadata& PointCloud::GetMetadata() const
{
  return metadata_;
}

Rank1View<const Real, HostMemorySpace> PointCloud::GetDOFHolderDataHost() const
{
  Kokkos::deep_copy(data_host_, data_);
  return make_const_array_view(data_host_);
}

void PointCloud::SetDOFHolderDataHost(
  Rank1View<const Real, HostMemorySpace> data)
{
  PCMS_FUNCTION_TIMER;
  PCMS_ALWAYS_ASSERT(data.size() == data_.size());
  Kokkos::parallel_for(
    Kokkos::RangePolicy<pcms::HostMemorySpace::execution_space>(0, data.size()),
    KOKKOS_CLASS_LAMBDA(int i) { data_host_(i) = data[i]; });
  Kokkos::deep_copy(data_, data_host_);
}

} // namespace pcms
