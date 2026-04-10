#ifndef PCMS_UNIFORM_GRID_FIELD_LAYOUT_H
#define PCMS_UNIFORM_GRID_FIELD_LAYOUT_H

#include "pcms/utility/arrays.h"
#include "pcms/discretization/discretization/uniform_grid.hpp"
#include "pcms/field/field_layout.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/field/field.h"
#include "pcms/utility/uniform_grid.h"

#include <array>

namespace pcms
{
template <unsigned Dim = 2>
class UniformGridFieldLayout : public FieldLayout
{
public:
  UniformGridFieldLayout(UniformGrid<Dim> grid, int num_components,
                         CoordinateSystem coordinate_system, int order = 1);

  std::shared_ptr<const Discretization>
  GetDiscretization() const noexcept override;

  int GetNumComponents() const override;
  LO GetNumOwnedDofHolder() const override;
  GO GetNumGlobalDofHolder() const override;

  Rank1View<const bool, HostMemorySpace> GetOwned() const override;
  GlobalIDView<HostMemorySpace> GetGids() const override;
  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates() const override;

  [[nodiscard]] bool IsDistributed() const override;

  EntOffsetsArray GetEntOffsets() const override;

  int GetDimension() const override;

  Rank1View<const LO, HostMemorySpace>
  GetDOFHolderClassificationDimensions() const override;

  Rank1View<const LO, HostMemorySpace>
  GetDOFHolderClassificationIds() const override;

  const UniformGrid<Dim>& GetGrid() const;
  LO GetNumCells() const;
  LO GetNumVertices() const;
  int GetOrder() const;

private:
  LO GetNumDofHolders() const;

  UniformGrid<Dim> grid_;
  int num_components_;
  CoordinateSystem coordinate_system_;
  int order_;
  Kokkos::View<GO*, HostMemorySpace> gids_;
  Kokkos::View<Real**, HostMemorySpace> dof_holder_coords_host_;
  Kokkos::View<Real**, DeviceMemorySpace> dof_holder_coords_;
  Kokkos::View<Real**, Kokkos::LayoutRight, DeviceMemorySpace> dof_holder_coords_device_right_;
  Kokkos::View<bool*, HostMemorySpace> owned_;
  Kokkos::View<LO*, HostMemorySpace> classification_dims_host_;
  Kokkos::View<LO*, HostMemorySpace> classification_ids_host_;
  std::shared_ptr<const Discretization> discretization_;
};

using UniformGridFieldLayout2D = UniformGridFieldLayout<2>;

} // namespace pcms
#endif // PCMS_UNIFORM_GRID_FIELD_LAYOUT_H
