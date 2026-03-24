#include "pcms/field/lagrange_field_factory.h"
#include "pcms/configuration.h"
#include "pcms/utility/assert.h"
#ifdef PCMS_ENABLE_MESHFIELDS
#include "pcms/field/adapter/meshfields/mesh_fields_adapter_layout.h"
#include "pcms/field/adapter/meshfields/mesh_fields_evaluator_factory.h"
#include "pcms/field/adapter/meshfields/mesh_fields_field_data.h"
#endif
#include "pcms/field/adapter/omega_h/omega_h_lagrange_layout.h"
#include "pcms/field/adapter/omega_h/omega_h_lagrange_evaluator_factory.h"
#include "pcms/field/adapter/uniform_grid/uniform_grid_field_layout.h"
#include "pcms/field/adapter/uniform_grid/uniform_grid_evaluator_factory.h"
#include "pcms/utility/uniform_grid.h"

#include <stdexcept>

namespace pcms
{

LagrangeFunctionSpace::LagrangeFunctionSpace(
  std::shared_ptr<const FieldLayout> layout,
  std::function<std::unique_ptr<FieldData<Real>>(FieldMetadata)>
    create_field_data_fn,
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept
  : layout_(std::move(layout)),
    create_field_data_fn_(std::move(create_field_data_fn)),
    evaluator_factory_(std::move(evaluator_factory))
{
}

LagrangeFunctionSpace LagrangeFunctionSpace::FromMesh(
  Omega_h::Mesh& mesh, int order, int num_components,
  CoordinateSystem coordinate_system, std::string global_id_name,
  Backend backend)
{
  // https://github.com/SCOREC/meshFields/issues/88
  if (backend == Backend::MeshFields && order == 0) {
    // MeshFields does not support order-0 fields; fall back to OmegaH.
    backend = Backend::OmegaH;
  }
  if (backend == Backend::MeshFields) {
#ifdef PCMS_ENABLE_MESHFIELDS
    if (num_components != 1) {
      throw pcms_error(
        "LagrangeFunctionSpace::FromMesh: MeshFields backend only supports "
        "single-component fields");
    }
    std::array<int, 4> nodes_per_dim{};
    switch (order) {
      case 1: nodes_per_dim = {1, 0, 0, 0}; break;
      case 2: nodes_per_dim = {1, 1, 0, 0}; break;
      default: throw pcms_error("Unimplemented Lagrange order");
    }
    auto mesh_layout = std::make_shared<MeshFieldsAdapterLayout>(
      mesh, nodes_per_dim, num_components, coordinate_system,
      std::move(global_id_name));
    auto eval_factory =
      std::make_shared<MeshFieldsEvaluatorFactory<Real>>(mesh_layout);
    return LagrangeFunctionSpace(
      mesh_layout,
      [mesh_layout](FieldMetadata metadata) {
        return std::make_unique<MeshFieldsFieldData<Real>>(mesh_layout,
                                                           metadata);
      },
      std::move(eval_factory));
#else
    throw pcms_error(
      "LagrangeFunctionSpace::FromMesh: MeshFields backend requested but "
      "PCMS_ENABLE_MESHFIELDS is not set");
#endif
  }
  // OmegaH backend
  if (order > 1) {
    throw pcms_error(
      "LagrangeFunctionSpace::FromMesh: OmegaH backend only supports "
      "Lagrange orders 0 and 1");
  }
  auto layout = std::make_shared<OmegaHLagrangeLayout>(
    mesh, order, num_components, coordinate_system, std::move(global_id_name));
  auto eval_factory =
    std::make_shared<OmegaHLagrangeEvaluatorFactory<Real>>(layout);
  return LagrangeFunctionSpace(
    layout,
    [layout](FieldMetadata metadata) {
      return std::make_unique<SimpleFieldData<Real>>(layout, metadata);
    },
    std::move(eval_factory));
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

LagrangeFunctionSpace LagrangeFunctionSpace::FromUniformGrid(
  Rank1View<Real, HostMemorySpace> edge_length,
  Rank1View<Real, HostMemorySpace> bot_left,
  Rank1View<LO, HostMemorySpace> divisions,
  int num_components,
  CoordinateSystem coordinate_system,
  int order)
{
  if (order != 0 && order != 1) {
    throw std::invalid_argument(
      "LagrangeFunctionSpace::FromUniformGrid: only orders 0 and 1 are supported");
  }
  if (edge_length.size() != bot_left.size() ||
      edge_length.size() != divisions.size()) {
    throw std::invalid_argument(
      "edge_length, bot_left, and divisions must have the same size");
  }
  switch (edge_length.size()) {
  case 2: {
    auto ug_layout = std::make_shared<UniformGridFieldLayout<2>>(
      MakeUniformGrid<2>(edge_length, bot_left, divisions), num_components,
      coordinate_system, order);
    auto eval_factory =
      std::make_shared<UniformGridEvaluatorFactory<2>>(ug_layout);
    return LagrangeFunctionSpace(
      ug_layout,
      [ug_layout](FieldMetadata metadata) {
        return std::make_unique<SimpleFieldData<Real>>(ug_layout, metadata);
      },
      std::move(eval_factory));
  }
  case 3: {
    auto ug_layout = std::make_shared<UniformGridFieldLayout<3>>(
      MakeUniformGrid<3>(edge_length, bot_left, divisions), num_components,
      coordinate_system, order);
    auto eval_factory =
      std::make_shared<UniformGridEvaluatorFactory<3>>(ug_layout);
    return LagrangeFunctionSpace(
      ug_layout,
      [ug_layout](FieldMetadata metadata) {
        return std::make_unique<SimpleFieldData<Real>>(ug_layout, metadata);
      },
      std::move(eval_factory));
  }
  default:
    throw std::invalid_argument(
      "LagrangeFunctionSpace::FromUniformGrid: only dim 2 and 3 are supported");
  }
}

std::shared_ptr<const FieldLayout>
LagrangeFunctionSpace::GetLayout() const noexcept
{
  return layout_;
}

CoordinateSystem LagrangeFunctionSpace::GetCoordinateSystem() const noexcept
{
  return evaluator_factory_->GetCoordinateSystem();
}

const FieldEvaluatorFactory<Real>&
LagrangeFunctionSpace::GetEvaluatorFactory() const noexcept
{
  return *evaluator_factory_;
}

std::unique_ptr<PointEvaluator<Real>> LagrangeFunctionSpace::CreatePointEvaluator(
  CoordinateView<HostMemorySpace> coords,
  OutOfBoundsPolicy policy) const
{
  if (!evaluator_factory_) {
    throw pcms_error(
      "LagrangeFunctionSpace::CreatePointEvaluator: evaluator construction is "
      "not available for this backend");
  }
  return evaluator_factory_->CreatePointEvaluator(coords, policy);
}

Field<Real> LagrangeFunctionSpace::CreateField(FieldMetadata metadata) const
{
  return Field<Real>(evaluator_factory_, create_field_data_fn_(metadata));
}

std::unique_ptr<FieldData<Real>> LagrangeFunctionSpace::CreateFieldData(
  FieldMetadata metadata) const
{
  return create_field_data_fn_(metadata);
}

} // namespace pcms
