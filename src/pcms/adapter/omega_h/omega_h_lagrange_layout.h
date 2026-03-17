#ifndef PCMS_ADAPTER_OMEGA_H_LAGRANGE_LAYOUT_H
#define PCMS_ADAPTER_OMEGA_H_LAGRANGE_LAYOUT_H

#include <Omega_h_mesh.hpp>
#include "pcms/utility/arrays.h"
#include "pcms/field_layout.h"
#include "pcms/coordinate_system.h"
#include "pcms/field.h"

namespace pcms
{

// Layout for native Omega_h Lagrange fields.
//   order 0: one DOF holder per element (centroid coordinates)
//   order 1: one DOF holder per vertex (barycentric interpolation)
// Throws std::invalid_argument for any other order.
class OmegaHLagrangeLayout : public FieldLayout
{
public:
  OmegaHLagrangeLayout(Omega_h::Mesh& mesh, int order, int num_components,
                       CoordinateSystem coordinate_system,
                       std::string global_id_name = "global");

  int GetNumComponents() const override;
  LO GetNumOwnedDofHolder() const override;
  GO GetNumGlobalDofHolder() const override;

  Rank1View<const bool, HostMemorySpace> GetOwned() const override;
  GlobalIDView<HostMemorySpace> GetGids() const override;
  CoordinateView<HostMemorySpace> GetDOFHolderCoordinates() const override;

  bool IsDistributed() override;

  EntOffsetsArray GetEntOffsets() const override;

  ReversePartitionMap2 GetReversePartitionMap(
    const redev::Partition& partition) const override;

  int GetOrder() const;
  Omega_h::Mesh& GetMesh() const;

private:
  Omega_h::Mesh& mesh_;
  int order_;
  int num_components_;
  CoordinateSystem coordinate_system_;
  std::string global_id_name_;

  Omega_h::HostWrite<Omega_h::GO> gids_host_;
  Omega_h::HostRead<Real> coords_host_; // flat row-major: [n_dofs * mesh_dim]
  Kokkos::View<bool*, HostMemorySpace> owned_host_;
  Omega_h::HostRead<Omega_h::ClassId> class_ids_host_;
  Omega_h::HostRead<Omega_h::I8> class_dims_host_;
};

} // namespace pcms
#endif // PCMS_ADAPTER_OMEGA_H_LAGRANGE_LAYOUT_H
