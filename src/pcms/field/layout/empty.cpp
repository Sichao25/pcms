#include "pcms/field/layout/empty.h"

namespace pcms
{

EmptyFieldLayout::EmptyFieldLayout()
  : owned_("null_owned", 0),
    gids_("null_gids", 0),
    class_dims_("null_class_dims", 0),
    class_ids_("null_class_ids", 0),
    coords_host_("null_coords_host", 0, 2),
    coords_("null_coords", 0, 2)
{
  discretization_ = std::make_shared<EmptyDiscretization>();
}

std::shared_ptr<const Discretization>
EmptyFieldLayout::GetDiscretization() const noexcept
{
  return discretization_;
}

int EmptyFieldLayout::GetNumComponents() const
{
  return 1;
}

LO EmptyFieldLayout::GetNumOwnedDofHolder() const
{
  return 0;
}

GO EmptyFieldLayout::GetNumGlobalDofHolder() const
{
  return 0;
}

Rank1View<const bool, HostMemorySpace> EmptyFieldLayout::GetOwned() const
{
  return make_const_array_view(owned_);
}

GlobalIDView<HostMemorySpace> EmptyFieldLayout::GetGids() const
{
  return make_const_array_view(gids_);
}

bool EmptyFieldLayout::IsDistributed() const
{
  return false;
}

EntOffsetsArray EmptyFieldLayout::GetEntOffsets() const
{
  return {0, 0, 0, 0, 0};
}

CoordinateView<DeviceMemorySpace> EmptyFieldLayout::GetDOFHolderCoordinates()
  const
{
  return {CoordinateSystem::XGC,
          Rank2View<const Real, DeviceMemorySpace>(coords_.data(), 0, 2)};
}

int EmptyFieldLayout::GetDimension() const
{
  return 2;
}

Rank1View<const LO, HostMemorySpace>
EmptyFieldLayout::GetDOFHolderClassificationDimensions() const
{
  return make_const_array_view(class_dims_);
}

Rank1View<const LO, HostMemorySpace>
EmptyFieldLayout::GetDOFHolderClassificationIds() const
{
  return make_const_array_view(class_ids_);
}

} // namespace pcms
