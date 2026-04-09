#include "pcms/field/layout/omega_h_entity.h"

#include "pcms/utility/assert.h"
#include "pcms/utility/mesh_geometry.h"

namespace pcms
{

namespace
{

template <typename T>
Omega_h::Write<Omega_h::GO> GetGidsHelper(Omega_h::Mesh& mesh,
                                          int entity_dim,
                                          const std::string& global_id_name)
{
  auto dim_gids = mesh.get_array<T>(entity_dim, global_id_name);
  Omega_h::Write<Omega_h::GO> gids(dim_gids.size());
  Omega_h::parallel_for(
    dim_gids.size(),
    OMEGA_H_LAMBDA(int i) { gids[i] = dim_gids[i]; });
  return gids;
}

Omega_h::HostWrite<Omega_h::GO> BuildGids(Omega_h::Mesh& mesh,
                                          int entity_dim,
                                          const std::string& global_id_name)
{
  auto tag = mesh.get_tagbase(entity_dim, global_id_name);
  Omega_h::Write<Omega_h::GO> gids;
  if (Omega_h::is<Omega_h::GO>(tag)) {
    gids = GetGidsHelper<Omega_h::GO>(mesh, entity_dim, global_id_name);
  } else if (Omega_h::is<Omega_h::LO>(tag)) {
    gids = GetGidsHelper<Omega_h::LO>(mesh, entity_dim, global_id_name);
  } else {
    std::cerr << "Weird tag type for global arrays.\n";
    std::abort();
  }
  return Omega_h::HostWrite<Omega_h::GO>(gids);
}

Kokkos::View<bool*, HostMemorySpace> BuildOwned(Omega_h::Mesh& mesh,
                                                int entity_dim)
{
  Kokkos::View<bool*, HostMemorySpace> owned("owned", mesh.nents(entity_dim));
  auto owned_h = Omega_h::HostRead<Omega_h::I8>(mesh.owned(entity_dim));
  for (int i = 0; i < mesh.nents(entity_dim); ++i)
    owned(i) = static_cast<bool>(owned_h[i]);
  return owned;
}

} // namespace

OmegaHEntityLayout::OmegaHEntityLayout(Omega_h::Mesh& mesh, int entity_dim,
                                       int num_components,
                                       CoordinateSystem coordinate_system,
                                       std::string global_id_name)
  : dimension_(mesh.dim()),
    entity_dim_(entity_dim),
    num_components_(num_components),
    num_global_dof_holder_(mesh.nglobal_ents(entity_dim)),
    coordinate_system_(coordinate_system),
    gids_host_(BuildGids(mesh, entity_dim, global_id_name)),
    coords_host_(get_entity_centroids(mesh, entity_dim)),
    coords_(get_entity_centroids(mesh, entity_dim)),
    class_ids_host_(mesh.get_array<Omega_h::ClassId>(entity_dim, "class_id")),
    class_dims_host_(mesh.get_array<Omega_h::I8>(entity_dim, "class_dim")),
    owned_host_(BuildOwned(mesh, entity_dim)),
    classification_dims_host_("classification_dims", mesh.nents(entity_dim)),
    classification_ids_host_("classification_ids", mesh.nents(entity_dim)),
    discretization_(std::make_shared<OmegaHDiscretization>(mesh))
{
  PCMS_ALWAYS_ASSERT(entity_dim_ >= 0 && entity_dim_ <= dimension_);

  for (int i = 0; i < mesh.nents(entity_dim_); ++i) {
    classification_dims_host_(i) = static_cast<LO>(class_dims_host_[i]);
    classification_ids_host_(i) = static_cast<LO>(class_ids_host_[i]);
  }
}

std::shared_ptr<const Discretization>
OmegaHEntityLayout::GetDiscretization() const noexcept
{
  return discretization_;
}

int OmegaHEntityLayout::GetNumComponents() const
{
  return num_components_;
}

LO OmegaHEntityLayout::GetNumOwnedDofHolder() const
{
  return static_cast<LO>(coords_host_.size() / dimension_);
}

GO OmegaHEntityLayout::GetNumGlobalDofHolder() const
{
  return num_global_dof_holder_;
}

Rank1View<const bool, HostMemorySpace> OmegaHEntityLayout::GetOwned() const
{
  return make_const_array_view(owned_host_);
}

GlobalIDView<HostMemorySpace> OmegaHEntityLayout::GetGids() const
{
  return GlobalIDView<HostMemorySpace>(gids_host_.data(), gids_host_.size());
}

CoordinateView<HostMemorySpace> OmegaHEntityLayout::GetDOFHolderCoordinatesHost() const
{
  Rank2View<const Real, HostMemorySpace> coords_view(
    coords_host_.data(), GetNumOwnedDofHolder(), dimension_);
  return CoordinateView<HostMemorySpace>{coordinate_system_, coords_view};
}

CoordinateView<DeviceMemorySpace> OmegaHEntityLayout::GetDOFHolderCoordinates() const
{
  Rank2View<const Real, DeviceMemorySpace> coords_view(
    coords_.data(), GetNumOwnedDofHolder(), dimension_);
  return CoordinateView<DeviceMemorySpace>{coordinate_system_, coords_view};
}

bool OmegaHEntityLayout::IsDistributed() const
{
  return true;
}

EntOffsetsArray OmegaHEntityLayout::GetEntOffsets() const
{
  EntOffsetsArray offsets{};
  offsets.fill(0);
  const auto n = static_cast<size_t>(GetNumOwnedDofHolder());
  for (int i = entity_dim_ + 1; i < ent_offsets_len; ++i)
    offsets[i] = n;
  return offsets;
}

int OmegaHEntityLayout::GetDimension() const
{
  return dimension_;
}

Rank1View<const LO, HostMemorySpace>
OmegaHEntityLayout::GetDOFHolderClassificationDimensions() const
{
  return make_const_array_view(classification_dims_host_);
}

Rank1View<const LO, HostMemorySpace>
OmegaHEntityLayout::GetDOFHolderClassificationIds() const
{
  return make_const_array_view(classification_ids_host_);
}

} // namespace pcms
