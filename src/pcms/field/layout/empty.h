#ifndef PCMS_EMPTY_FIELD_LAYOUT_H
#define PCMS_EMPTY_FIELD_LAYOUT_H

#include "pcms/discretization/discretization/empty.hpp"
#include "pcms/field/field_layout.h"
#include <Kokkos_Core.hpp>

namespace pcms
{

class EmptyFieldLayout : public FieldLayout
{
public:
  EmptyFieldLayout();

  std::shared_ptr<const Discretization>
  GetDiscretization() const noexcept override;

  int GetNumComponents() const override;
  LO GetNumOwnedDofHolder() const override;
  GO GetNumGlobalDofHolder() const override;
  Rank1View<const bool, HostMemorySpace> GetOwned() const override;
  GlobalIDView<HostMemorySpace> GetGids() const override;
  bool IsDistributed() const override;
  EntOffsetsArray GetEntOffsets() const override;
  CoordinateView<HostMemorySpace> GetDOFHolderCoordinates() const override;
  int GetDimension() const override;
  Rank1View<const LO, HostMemorySpace>
  GetDOFHolderClassificationDimensions() const override;
  Rank1View<const LO, HostMemorySpace>
  GetDOFHolderClassificationIds() const override;

private:
  Kokkos::View<bool*, HostMemorySpace> owned_;
  Kokkos::View<GO*, HostMemorySpace> gids_;
  Kokkos::View<LO*, HostMemorySpace> class_dims_;
  Kokkos::View<LO*, HostMemorySpace> class_ids_;
  Kokkos::View<Real**, HostMemorySpace> coords_;
  std::shared_ptr<const Discretization> discretization_;
};

} // namespace pcms

#endif // PCMS_EMPTY_FIELD_LAYOUT_H
