#include "pcms/field/function_space/nodal.h"
#include "pcms/field/layout/point_cloud.h"
#include "pcms/field/evaluator/point_cloud.h"
#include "pcms/field/data/simple.h"
#include "pcms/discretization/discretization/omega_h.hpp"
#include "pcms/utility/assert.h"
#include "pcms/utility/common.h"
#include "pcms/utility/mesh_geometry.h"
#include "pcms/localization/point_cloud_localization.h"
#include "pcms/localization/mesh_localization.h"

#include <Kokkos_Core.hpp>
#include <Omega_h_array.hpp>

namespace pcms
{

NodalFunctionSpace::NodalFunctionSpace(
  std::shared_ptr<const FieldLayout> layout,
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept
  : layout_(std::move(layout)),
    evaluator_factory_(std::move(evaluator_factory))
{
}

NodalFunctionSpace NodalFunctionSpace::Create(
  Rank2View<Real, HostMemorySpace> coords,
  CoordinateSystem coordinate_system,
  MLSOptions options)
{
  int dim = static_cast<int>(coords.extent(1));
  Kokkos::View<Real**, Kokkos::HostSpace> host_view(
    coords.data_handle(), coords.extent(0), coords.extent(1));
  auto device_view =
    Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace{}, host_view);
  auto pc_layout =
    std::make_shared<PointCloudLayout>(dim, device_view, coordinate_system);
  auto localization = std::make_shared<PointCloudLocalizationFactory>(pc_layout, options);
  auto eval_factory =
    std::make_shared<PointCloudEvaluatorFactory>(pc_layout, localization, options);
  return NodalFunctionSpace(pc_layout, std::move(eval_factory));
}

NodalFunctionSpace NodalFunctionSpace::FromMesh(
  Omega_h::Mesh& mesh,
  int source_entity_dim,
  CoordinateSystem coordinate_system,
  MLSOptions options)
{
  if (coordinate_system != CoordinateSystem::Cartesian) {
    throw pcms_error(
      "NodalFunctionSpace::FromMesh: only Cartesian coordinates are "
      "currently supported for MLS");
  }
  if (source_entity_dim < 0 || source_entity_dim > mesh.dim()) {
    throw pcms_error(
      "NodalFunctionSpace::FromMesh: source_entity_dim is out of range");
  }

  const Omega_h::Reals source_coords_oh = get_entity_centroids(mesh, source_entity_dim);

  const int dim = mesh.dim();
  const int n_src = static_cast<int>(source_coords_oh.size()) / dim;

  Omega_h::HostRead<Omega_h::Real> src_read(source_coords_oh);
  Kokkos::View<Real**, Kokkos::HostSpace> coords_host("pc_coords", n_src, dim);
  for (int i = 0; i < n_src; ++i)
    for (int d = 0; d < dim; ++d)
      coords_host(i, d) = src_read[i * dim + d];
  auto coords_dev = Kokkos::create_mirror_view_and_copy(
    Kokkos::DefaultExecutionSpace{}, coords_host);

  auto discretization = std::make_shared<OmegaHDiscretization>(mesh);
  auto pc_layout = std::make_shared<PointCloudLayout>(
    dim, coords_dev, coordinate_system, discretization, source_entity_dim);
  auto localization =
    std::make_shared<AdjacencyLocalizationFactory>(mesh, source_entity_dim, options);
  auto eval_factory =
    std::make_shared<PointCloudEvaluatorFactory>(pc_layout, localization, options);
  return NodalFunctionSpace(pc_layout, std::move(eval_factory));
}

std::shared_ptr<const FieldLayout>
NodalFunctionSpace::GetLayout() const noexcept
{
  return layout_;
}

CoordinateSystem NodalFunctionSpace::GetCoordinateSystem() const noexcept
{
  return evaluator_factory_->GetCoordinateSystem();
}

FieldVariant NodalFunctionSpace::CreateFieldImpl(
  Type value_type, FieldMetadata metadata) const
{
  return apply_to_type(value_type, [&](auto tag) -> FieldVariant {
    using T = typename decltype(tag)::type;
    if constexpr (!std::is_same_v<T, double>) {
      throw pcms_error("NodalFunctionSpace: only double (Real) is supported");
    }
    else {
      return WrapField<double>(
        layout_, std::make_unique<SimpleFieldData<double>>(layout_, metadata),
        evaluator_factory_);
    }
  });
}

FieldVariant NodalFunctionSpace::CreateFieldImpl(FieldDataVariant data) const
{
  if (!std::holds_alternative<std::unique_ptr<FieldData<double>>>(data)) {
    throw pcms_error("NodalFunctionSpace: only double (Real) is supported");
  }
  auto fd = std::move(std::get<std::unique_ptr<FieldData<double>>>(data));
  PCMS_ALWAYS_ASSERT(fd != nullptr);
  if (dynamic_cast<const SimpleFieldData<double>*>(fd.get()) == nullptr) {
    throw pcms_error(
      "NodalFunctionSpace::CreateField: requires SimpleFieldData<double>");
  }
  if (fd->GetDOFHolderDataHost().size() !=
      detail::ExpectedFlatFieldDataSize(*layout_)) {
    throw pcms_error(
      "NodalFunctionSpace::CreateField: field data size does not match "
      "layout");
  }
  return WrapField<double>(layout_, std::move(fd), evaluator_factory_);
}

PointEvaluatorVariant NodalFunctionSpace::CreatePointEvaluatorImpl(
  Type value_type,
  const EvaluationRequest& request) const
{
  if (value_type != Type::Real) {
    throw pcms_error(
      "NodalFunctionSpace: point evaluation only supports double (Real)");
  }
  PCMS_ALWAYS_ASSERT(evaluator_factory_ != nullptr);
  return evaluator_factory_->CreatePointEvaluator(request);
}

} // namespace pcms
