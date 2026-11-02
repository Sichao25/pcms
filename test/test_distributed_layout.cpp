// Checks the owned/local split of a FieldLayout on a ghosted, distributed
// Omega_h mesh: local = owned + ghost DOF holders, and the compact owned views
// (owned->local map, owned GIDs, owned coordinates, owned field data) agree
// with the local ones.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include "pcms/configuration.h"
#include "pcms/field/coordinate_systems/cartesian.hpp"
#include "pcms/field/function_space/lagrange.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/types.h"

#include <cstdio>
#include <string>
#include <vector>

using pcms::GO;
using pcms::LO;
using pcms::Real;

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "[rank %d] %s\n", rank, message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

template <typename View>
auto ToHost(const View& view)
{
  return Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), view);
}

void CheckLayout(Omega_h::Mesh& mesh, const pcms::LagrangeFunctionSpace& space,
                 const std::string& label)
{
  const auto& layout = *space.GetLayout();
  const LO n_local = layout.GetNumLocalDofHolder();
  const LO n_owned = layout.GetNumOwnedDofHolder();
  const int nc = layout.GetNumComponents();
  const int dim = layout.GetDimension();

  require(layout.IsDistributed(), label + ": layout is not distributed");
  require(n_local == mesh.nverts(), label + ": local count != mesh vertices");
  require(n_owned <= n_local, label + ": more owned than local holders");
  require(layout.LocalSize() == n_local * nc, label + ": LocalSize mismatch");
  require(layout.OwnedSize() == n_owned * nc, label + ": OwnedSize mismatch");
  require(layout.OwnedIsLocal() == (n_owned == n_local),
          label + ": OwnedIsLocal mismatch");

  // Every holder is owned by exactly one rank, and a ghosted mesh has ghosts.
  long long owned_sum = n_owned;
  long long ghost_sum = n_local - n_owned;
  MPI_Allreduce(MPI_IN_PLACE, &owned_sum, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ghost_sum, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  require(owned_sum == static_cast<long long>(layout.GetNumGlobalDofHolder()),
          label + ": owned counts do not sum to the global count");
  require(ghost_sum > 0, label + ": ghosted mesh produced no ghost holders");

  // owned->local map: one entry per owned holder, increasing, owned only.
  const auto owned = layout.GetOwnedHost();
  const auto owned_to_local = layout.GetOwnedToLocalHost();
  require(static_cast<LO>(owned_to_local.size()) == n_owned,
          label + ": owned->local map has the wrong size");
  LO n_owned_flags = 0;
  for (LO i = 0; i < n_local; ++i) {
    n_owned_flags += owned[i] ? 1 : 0;
  }
  require(n_owned_flags == n_owned, label + ": owned flags != owned count");
  for (LO o = 0; o < n_owned; ++o) {
    const LO l = owned_to_local(o);
    require(l >= 0 && l < n_local && owned[l],
            label + ": owned->local maps to a non-owned holder");
    require(o == 0 || owned_to_local(o - 1) < l,
            label + ": owned->local map is not increasing");
  }
  const auto owned_to_local_device = ToHost(layout.GetOwnedToLocal());
  require(static_cast<LO>(owned_to_local_device.size()) == n_owned,
          label + ": device owned->local map has the wrong size");
  for (LO o = 0; o < n_owned; ++o) {
    require(owned_to_local_device(o) == owned_to_local(o),
            label + ": device and host owned->local maps differ");
  }

  // Owned GIDs and coordinates are the local ones, gathered.
  const auto gids = layout.GetGidsHost();
  const auto owned_gids = layout.GetOwnedGidsHost();
  const auto gids_device = layout.GetGids();
  const auto owned_gids_device = layout.GetOwnedGids();
  require(static_cast<LO>(gids_device.size()) == n_local,
          label + ": device GIDs have the wrong size");
  require(static_cast<LO>(owned_gids.size()) == n_owned &&
            static_cast<LO>(owned_gids_device.size()) == n_owned,
          label + ": owned GIDs have the wrong size");
  Kokkos::View<GO*, pcms::HostMemorySpace> gids_device_host("gids", n_local);
  Kokkos::deep_copy(
    gids_device_host,
    Kokkos::View<const GO*, pcms::DeviceMemorySpace, Kokkos::MemoryUnmanaged>(
      gids_device.data_handle(), n_local));
  Kokkos::View<GO*, pcms::HostMemorySpace> owned_gids_device_host("owned_gids",
                                                                  n_owned);
  Kokkos::deep_copy(
    owned_gids_device_host,
    Kokkos::View<const GO*, pcms::DeviceMemorySpace, Kokkos::MemoryUnmanaged>(
      owned_gids_device.data_handle(), n_owned));
  for (LO i = 0; i < n_local; ++i) {
    require(gids_device_host(i) == gids[i],
            label + ": device and host GIDs differ");
  }
  for (LO o = 0; o < n_owned; ++o) {
    require(owned_gids[o] == gids[owned_to_local(o)],
            label + ": owned GID != local GID at the mapped index");
    require(owned_gids_device_host(o) == owned_gids[o],
            label + ": device and host owned GIDs differ");
  }

  const auto coords = layout.GetDOFHolderCoordinates().GetValues();
  const auto owned_coords = layout.GetOwnedDOFHolderCoordinates().GetValues();
  require(static_cast<LO>(owned_coords.extent(0)) == n_owned,
          label + ": owned coordinates have the wrong size");
  Kokkos::View<Real**, pcms::DeviceMemorySpace> coords_copy("coords", n_local,
                                                            dim);
  Kokkos::View<Real**, pcms::DeviceMemorySpace> owned_coords_copy(
    "owned_coords", n_owned, dim);
  pcms::CopyDeviceRank2ViewToRank2View(pcms::MakeRank2View(coords_copy),
                                       coords);
  pcms::CopyDeviceRank2ViewToRank2View(pcms::MakeRank2View(owned_coords_copy),
                                       owned_coords);
  const auto coords_host = ToHost(coords_copy);
  const auto owned_coords_host = ToHost(owned_coords_copy);
  for (LO o = 0; o < n_owned; ++o) {
    for (int d = 0; d < dim; ++d) {
      require(owned_coords_host(o, d) == coords_host(owned_to_local(o), d),
              label + ": owned coordinate != local coordinate");
    }
  }

  // Owned entity offsets cover exactly the owned holders.
  const auto owned_offsets = layout.GetOwnedEntOffsets();
  require(owned_offsets.back() == static_cast<size_t>(n_owned),
          label + ": owned entity offsets do not end at the owned count");

  // Field data: write all local holders, read back the compact owned subset.
  auto field = space.CreateFunction<Real>();
  std::vector<Real> values(static_cast<size_t>(n_local) * nc);
  for (LO i = 0; i < n_local; ++i) {
    for (int c = 0; c < nc; ++c) {
      values[static_cast<size_t>(i) * nc + c] =
        static_cast<Real>(gids[i]) + 0.25 * c;
    }
  }
  field.SetDOFHolderDataUncheckedHost(
    pcms::Rank2View<const Real, pcms::HostMemorySpace>(values.data(), n_local,
                                                       nc));

  const auto owned_host = field.GetOwnedDOFHolderDataHost().GetValues();
  require(static_cast<LO>(owned_host.extent(0)) == n_owned &&
            static_cast<int>(owned_host.extent(1)) == nc,
          label + ": owned host data has the wrong shape");
  const auto owned_device = field.GetOwnedDOFHolderData().GetValues();
  require(static_cast<LO>(owned_device.extent(0)) == n_owned &&
            static_cast<int>(owned_device.extent(1)) == nc,
          label + ": owned device data has the wrong shape");
  Kokkos::View<Real**, pcms::DeviceMemorySpace> owned_device_copy(
    "owned_device", n_owned, nc);
  pcms::CopyDeviceRank2ViewToRank2View(pcms::MakeRank2View(owned_device_copy),
                                       owned_device);
  const auto owned_device_host = ToHost(owned_device_copy);
  for (LO o = 0; o < n_owned; ++o) {
    for (int c = 0; c < nc; ++c) {
      const Real expected = static_cast<Real>(owned_gids[o]) + 0.25 * c;
      require(owned_host(o, c) == expected,
              label + ": owned host data != local data at the mapped index");
      require(owned_device_host(o, c) == expected,
              label + ": owned device data != local data at the mapped index");
    }
  }
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  {
    Omega_h::Library library(&argc, &argv);
    auto world = library.world();
    require(world->size() >= 2, "test requires at least 2 MPI ranks");

    const int n = 4 * world->size();
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, n, n, 0, false);
    mesh.set_parting(OMEGA_H_GHOSTED, 1, false);

    for (int nc : {1, 2}) {
      const std::string suffix = " (" + std::to_string(nc) + " components)";
      CheckLayout(mesh,
                  *pcms::LagrangeFunctionSpace::FromMesh(
                    mesh, 1, nc, pcms::csys::Cartesian::Deferred(), "global",
                    pcms::LagrangeFunctionSpace::Backend::OmegaH),
                  "Omega_h backend" + suffix);
#ifdef PCMS_ENABLE_MESHFIELDS
      CheckLayout(mesh,
                  *pcms::LagrangeFunctionSpace::FromMesh(
                    mesh, 1, nc, pcms::csys::Cartesian::Deferred(), "global",
                    pcms::LagrangeFunctionSpace::Backend::MeshFields),
                  "MeshFields backend" + suffix);
#endif
    }

    if (world->rank() == 0) {
      std::printf("distributed layout test passed\n");
    }
  }
  MPI_Finalize();
  return 0;
}
