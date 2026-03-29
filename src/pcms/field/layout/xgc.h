#ifndef PCMS_XGC_FIELD_LAYOUT_H
#define PCMS_XGC_FIELD_LAYOUT_H

#include "pcms/discretization/discretization/xgc.hpp"
#include "pcms/discretization/discretization/xgc_reverse_classification.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/field/field_layout.h"
#include <functional>

namespace pcms
{

class XGCFieldLayout : public FieldLayout
{
public:
  XGCFieldLayout(const ReverseClassificationVertex& reverse_classification,
                 std::function<int8_t(int, int)> in_overlap,
                 LO num_plane_nodes);

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

  LO GetFullDataSize() const noexcept;

private:
  Kokkos::View<bool*, HostMemorySpace> owned_;
  Kokkos::View<GO*, HostMemorySpace> gids_;
  Kokkos::View<LO*, HostMemorySpace> class_dims_;
  Kokkos::View<LO*, HostMemorySpace> class_ids_;
  Kokkos::View<Real**, HostMemorySpace> coords_;
  LO num_plane_nodes_;
  std::shared_ptr<const Discretization> discretization_;
};

} // namespace pcms

#endif // PCMS_XGC_FIELD_LAYOUT_H
