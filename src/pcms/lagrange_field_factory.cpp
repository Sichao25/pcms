#include "pcms/lagrange_field_factory.h"
#include "pcms/adapter/meshfields/mesh_fields_adapter_layout.h"
#include "pcms/adapter/meshfields/mesh_fields_adapter2.h"
#include "pcms/adapter/uniform_grid/uniform_grid_field_layout.h"
#include "pcms/adapter/uniform_grid/uniform_grid_field.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/uniform_grid.h"

#include <stdexcept>

namespace pcms
{

LagrangeFieldFactory::LagrangeFieldFactory(
  std::shared_ptr<const FieldLayout> layout,
  std::function<std::unique_ptr<FieldT<Real>>()> create_fn) noexcept
  : layout_(std::move(layout)), create_fn_(std::move(create_fn))
{
}

LagrangeFieldFactory LagrangeFieldFactory::FromMesh(
  Omega_h::Mesh& mesh, int order, int num_components,
  CoordinateSystem coordinate_system, std::string global_id_name)
{
  std::array<int, 4> nodes_per_dim{};
  switch (order) {
    case 1: nodes_per_dim = {1, 0, 0, 0}; break;
    case 2: nodes_per_dim = {1, 1, 0, 0}; break;
    default: throw std::runtime_error("Unimplemented Lagrange order");
  }
  auto mesh_layout = std::make_shared<MeshFieldsAdapterLayout>(
    mesh, nodes_per_dim, num_components, coordinate_system,
    std::move(global_id_name));
  return LagrangeFieldFactory(
    mesh_layout,
    [mesh_layout]() {
      return std::make_unique<MeshFieldsAdapter2<Real>>(mesh_layout);
    });
}

namespace
{
template <unsigned Dim>
UniformGrid<Dim> MakeUniformGrid(
  const Rank1View<Real, HostMemorySpace> edge_length,
  const Rank1View<Real, HostMemorySpace> bot_left,
  const Rank1View<LO, HostMemorySpace> divisions)
{
  PCMS_ALWAYS_ASSERT(edge_length.size() == Dim);
  PCMS_ALWAYS_ASSERT(bot_left.size() == Dim);
  PCMS_ALWAYS_ASSERT(divisions.size() == Dim);

  UniformGrid<Dim> grid;
  for (unsigned i = 0; i < Dim; ++i) {
    grid.edge_length[i] = edge_length[i];
    grid.bot_left[i] = bot_left[i];
    grid.divisions[i] = divisions[i];
  }
  return grid;
}
} // namespace

LagrangeFieldFactory LagrangeFieldFactory::FromUniformGrid(
  Rank1View<Real, HostMemorySpace> edge_length,
  Rank1View<Real, HostMemorySpace> bot_left,
  Rank1View<LO, HostMemorySpace> divisions,
  int num_components,
  CoordinateSystem coordinate_system)
{
  if (edge_length.size() != bot_left.size() ||
      edge_length.size() != divisions.size()) {
    throw std::invalid_argument(
      "edge_length, bot_left, and divisions must have the same size");
  }
  switch (edge_length.size()) {
  case 2: {
    auto ug_layout = std::make_shared<UniformGridFieldLayout<2>>(
      MakeUniformGrid<2>(edge_length, bot_left, divisions), num_components,
      coordinate_system);
    return LagrangeFieldFactory(
      ug_layout,
      [ug_layout]() {
        return std::make_unique<UniformGridField<2>>(ug_layout);
      });
  }
  case 3: {
    auto ug_layout = std::make_shared<UniformGridFieldLayout<3>>(
      MakeUniformGrid<3>(edge_length, bot_left, divisions), num_components,
      coordinate_system);
    return LagrangeFieldFactory(
      ug_layout,
      [ug_layout]() {
        return std::make_unique<UniformGridField<3>>(ug_layout);
      });
  }
  default:
    throw std::invalid_argument(
      "LagrangeFieldFactory::FromUniformGrid: only dim 2 and 3 are supported");
  }
}

std::shared_ptr<const FieldLayout>
LagrangeFieldFactory::GetLayout() const noexcept
{
  return layout_;
}

std::unique_ptr<FieldT<Real>> LagrangeFieldFactory::CreateFieldReal() const
{
  return create_fn_();
}

} // namespace pcms
