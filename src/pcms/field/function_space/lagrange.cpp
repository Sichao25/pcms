#include "pcms/field/function_space/lagrange.h"
#include "pcms/configuration.h"
#include "pcms/utility/assert.h"
#ifdef PCMS_ENABLE_MESHFIELDS
#include "pcms/field/layout/mesh_fields.h"
#include "pcms/field/evaluator/mesh_fields.h"
#include "pcms/field/data/mesh_fields.h"
#endif
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/field/evaluator/omega_h_lagrange.h"
#include "pcms/field/layout/uniform_grid.h"
#include "pcms/field/evaluator/uniform_grid.h"
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

LagrangeFunctionSpace LagrangeFunctionSpace::FromUniformGrid(
  const UniformGrid<2>& grid, int num_components,
  CoordinateSystem coordinate_system, int order)
{
  if (order != 0 && order != 1) {
    throw std::invalid_argument("LagrangeFunctionSpace::FromUniformGrid: only "
                                "orders 0 and 1 are supported");
  }
  auto ug_layout = std::make_shared<UniformGridFieldLayout<2>>(
    grid, num_components, coordinate_system, order);
  auto eval_factory =
    std::make_shared<UniformGridEvaluatorFactory<2>>(ug_layout);
  return LagrangeFunctionSpace(
    ug_layout,
    [ug_layout](FieldMetadata metadata) {
      return std::make_unique<SimpleFieldData<Real>>(ug_layout, metadata);
    },
    std::move(eval_factory));
}

LagrangeFunctionSpace LagrangeFunctionSpace::FromUniformGrid(
  const UniformGrid<3>& grid, int num_components,
  CoordinateSystem coordinate_system, int order)
{
  if (order != 0 && order != 1) {
    throw std::invalid_argument(
      "LagrangeFunctionSpace::FromUniformGrid: only orders 0 and 1 are supported");
  }
  auto ug_layout = std::make_shared<UniformGridFieldLayout<3>>(
    grid, num_components, coordinate_system, order);
  auto eval_factory =
    std::make_shared<UniformGridEvaluatorFactory<3>>(ug_layout);
  return LagrangeFunctionSpace(
    ug_layout,
    [ug_layout](FieldMetadata metadata) {
      return std::make_unique<SimpleFieldData<Real>>(ug_layout, metadata);
    },
    std::move(eval_factory));
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
LagrangeFunctionSpace::GetEvaluatorFactory() const
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
