#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/mesh_geometry.h"
#include "pcms/utility/profile.h"
#include <stdexcept>

namespace pcms
{

namespace
{
int EntityDimForOrder(int order, int mesh_dim)
{
  switch (order) {
    case 0: return mesh_dim; // one DOF per element
    case 1: return 0;        // one DOF per vertex
    default:
      throw std::invalid_argument(
        "OmegaHLagrangeLayout: only order 0 and 1 are supported");
  }
}

template <typename T>
Omega_h::Write<Omega_h::GO> GetGidsHelper(Omega_h::Mesh& mesh, int entity_dim,
                                          const std::string& global_id_name)
{
  auto dim_gids = mesh.get_array<T>(entity_dim, global_id_name);
  Omega_h::Write<Omega_h::GO> gids(dim_gids.size());
  Omega_h::parallel_for(
    dim_gids.size(),
    OMEGA_H_LAMBDA(int i) { gids[i] = dim_gids[i]; });
  return gids;
}

Omega_h::HostWrite<Omega_h::GO> BuildGids(Omega_h::Mesh& mesh, int entity_dim,
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
  int n = mesh.nents(entity_dim);
  Kokkos::View<bool*, HostMemorySpace> owned("owned", n);
  auto src = Omega_h::HostRead<Omega_h::I8>(mesh.owned(entity_dim));
  for (int i = 0; i < n; ++i)
    owned(i) = static_cast<bool>(src[i]);
  return owned;
}

Kokkos::View<bool*, HostMemorySpace> BuildOwned(Omega_h::Mesh& mesh,
                                                int entity_dim,
                                                Omega_h::Read<Omega_h::I8> mask)
{
  auto owned = BuildOwned(mesh, entity_dim);
  auto mask_h = Omega_h::HostRead<Omega_h::I8>(mask);
  PCMS_ALWAYS_ASSERT(static_cast<int>(mask_h.size()) == mesh.nents(entity_dim));
  for (int i = 0; i < mesh.nents(entity_dim); ++i) {
    owned(i) = owned(i) && static_cast<bool>(mask_h[i]);
  }
  return owned;
}

struct ConvertTo2DFunctor
{
  Kokkos::View<Real**, DeviceMemorySpace> coords_2d;
  Omega_h::Read<Real> coords;
  int dim;

  ConvertTo2DFunctor(Kokkos::View<Real**, DeviceMemorySpace> coords_2d_in,
                     Omega_h::Read<Real> coords_in, int dim_in)
    : coords_2d(coords_2d_in), coords(coords_in), dim(dim_in)
  {
  }

  KOKKOS_INLINE_FUNCTION void operator()(int i) const
  {
    int e = i / dim;
    int d = i % dim;
    coords_2d(e, d) = coords[i];
  }
};

} // namespace

OmegaHLagrangeLayout::OmegaHLagrangeLayout(Omega_h::Mesh& mesh, int order,
                                           int num_components,
                                           CoordinateSystem coordinate_system,
                                           std::string global_id_name)
  : mesh_(mesh),
    order_(order),
    num_components_(num_components),
    coordinate_system_(coordinate_system),
    global_id_name_(std::move(global_id_name))
{
  PCMS_FUNCTION_TIMER;
  int entity_dim = EntityDimForOrder(order_, mesh_.dim());

  gids_host_ = BuildGids(mesh_, entity_dim, global_id_name_);
  coords_ = get_entity_centroids(mesh_, entity_dim);
  coords_2d_ = Kokkos::View<Real**, DeviceMemorySpace>("coords_2d", mesh_.nents(entity_dim), mesh_.dim());
  int nents = mesh_.nents(entity_dim);
  int dim = mesh_.dim();
  ConvertTo2DFunctor functor(coords_2d_, coords_, dim);
  Kokkos::parallel_for("copy_coords_to_2d", nents * dim, functor);
  owned_host_ = BuildOwned(mesh_, entity_dim);

  class_ids_host_ = Omega_h::HostRead<Omega_h::ClassId>(
    mesh_.get_array<Omega_h::ClassId>(entity_dim, "class_id"));
  class_dims_host_ = Omega_h::HostRead<Omega_h::I8>(
    mesh_.get_array<Omega_h::I8>(entity_dim, "class_dim"));

  int n = mesh_.nents(entity_dim);
  classification_dims_host_ =
    Kokkos::View<LO*, HostMemorySpace>("classification_dims", n);
  classification_ids_host_ =
    Kokkos::View<LO*, HostMemorySpace>("classification_ids", n);
  for (int i = 0; i < n; ++i) {
    classification_dims_host_(i) = static_cast<LO>(class_dims_host_[i]);
    classification_ids_host_(i) = static_cast<LO>(class_ids_host_[i]);
  }
  discretization_ = std::make_shared<OmegaHDiscretization>(mesh_);
}

OmegaHLagrangeLayout::OmegaHLagrangeLayout(Omega_h::Mesh& mesh, int order,
                                           int num_components,
                                           CoordinateSystem coordinate_system,
                                           Omega_h::Read<Omega_h::I8> owned_mask,
                                           std::string global_id_name)
  : mesh_(mesh),
    order_(order),
    num_components_(num_components),
    coordinate_system_(coordinate_system),
    global_id_name_(std::move(global_id_name))
{
  PCMS_FUNCTION_TIMER;
  int entity_dim = EntityDimForOrder(order_, mesh_.dim());

  gids_host_ = BuildGids(mesh_, entity_dim, global_id_name_);
  coords_ = get_entity_centroids(mesh_, entity_dim);
  coords_2d_ = Kokkos::View<Real**, DeviceMemorySpace>("coords_2d", mesh_.nents(entity_dim), mesh_.dim());
  int nents = mesh_.nents(entity_dim);
  int dim = mesh_.dim();
  ConvertTo2DFunctor functor(coords_2d_, coords_, dim);
  Kokkos::parallel_for("copy_coords_to_2d", nents * dim, functor);
  owned_host_ = BuildOwned(mesh_, entity_dim, owned_mask);

  class_ids_host_ = Omega_h::HostRead<Omega_h::ClassId>(
    mesh_.get_array<Omega_h::ClassId>(entity_dim, "class_id"));
  class_dims_host_ = Omega_h::HostRead<Omega_h::I8>(
    mesh_.get_array<Omega_h::I8>(entity_dim, "class_dim"));

  int n = mesh_.nents(entity_dim);
  classification_dims_host_ =
    Kokkos::View<LO*, HostMemorySpace>("classification_dims", n);
  classification_ids_host_ =
    Kokkos::View<LO*, HostMemorySpace>("classification_ids", n);
  for (int i = 0; i < n; ++i) {
    classification_dims_host_(i) = static_cast<LO>(class_dims_host_[i]);
    classification_ids_host_(i) = static_cast<LO>(class_ids_host_[i]);
  }
  discretization_ = std::make_shared<OmegaHDiscretization>(mesh_);
}

std::shared_ptr<const Discretization>
OmegaHLagrangeLayout::GetDiscretization() const noexcept
{
  return discretization_;
}

int OmegaHLagrangeLayout::GetNumComponents() const
{
  return num_components_;
}

LO OmegaHLagrangeLayout::GetNumOwnedDofHolder() const
{
  return mesh_.nents(EntityDimForOrder(order_, mesh_.dim()));
}

GO OmegaHLagrangeLayout::GetNumGlobalDofHolder() const
{
  return mesh_.nglobal_ents(EntityDimForOrder(order_, mesh_.dim()));
}

Rank1View<const bool, HostMemorySpace> OmegaHLagrangeLayout::GetOwned() const
{
  return make_const_array_view(owned_host_);
}

GlobalIDView<HostMemorySpace> OmegaHLagrangeLayout::GetGids() const
{
  return GlobalIDView<HostMemorySpace>(gids_host_.data(), gids_host_.size());
}

CoordinateView<DeviceMemorySpace>
OmegaHLagrangeLayout::GetDOFHolderCoordinates() const
{
  int n = mesh_.nents(EntityDimForOrder(order_, mesh_.dim()));
  int dim = mesh_.dim();
  using LayoutPolicy = detail::default_layout_for_memory_space_t<DeviceMemorySpace>;
  Rank2View<const Real, DeviceMemorySpace, LayoutPolicy> coords_view(coords_2d_.data(), n, dim);
  return CoordinateView<DeviceMemorySpace, LayoutPolicy>{coordinate_system_, coords_view};
}

bool OmegaHLagrangeLayout::IsDistributed() const {
  return true;
}

EntOffsetsArray OmegaHLagrangeLayout::GetEntOffsets() const
{
  // Slot i holds the starting DOF index for entity dimension i.
  // Slot 4 holds the total DOF count.
  // For order-1 (entity_dim=0): offsets = {0, n, n, n, n}
  // For order-0 (entity_dim=mesh.dim()): offsets = {0,..,0, n, n}
  EntOffsetsArray offsets{};
  offsets.fill(0);
  int entity_dim = EntityDimForOrder(order_, mesh_.dim());
  LO n = mesh_.nents(entity_dim);
  for (int i = entity_dim + 1; i < ent_offsets_len; ++i)
    offsets[i] = n;
  return offsets;
}


int OmegaHLagrangeLayout::GetDimension() const
{
  return mesh_.dim();
}

Rank1View<const LO, HostMemorySpace>
OmegaHLagrangeLayout::GetDOFHolderClassificationDimensions() const
{
  return make_const_array_view(classification_dims_host_);
}

Rank1View<const LO, HostMemorySpace>
OmegaHLagrangeLayout::GetDOFHolderClassificationIds() const
{
  return make_const_array_view(classification_ids_host_);
}

int OmegaHLagrangeLayout::GetOrder() const
{
  return order_;
}

Omega_h::Mesh& OmegaHLagrangeLayout::GetMesh() const
{
  return mesh_;
}

} // namespace pcms
