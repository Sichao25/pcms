#ifndef PCMS_LAGRANGE_FIELD_FACTORY_H
#define PCMS_LAGRANGE_FIELD_FACTORY_H

#include <Omega_h_mesh.hpp>
#include "pcms/configuration.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/simple_field_data.h"
#include "coordinate_system.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/types.h"
#include "pcms/utility/memory_spaces.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace pcms
{

class LagrangeFieldFactory
{
public:
  enum class Backend { MeshFields, OmegaH };

#ifdef PCMS_ENABLE_MESHFIELDS
  static constexpr Backend DefaultBackend = Backend::MeshFields;
#else
  static constexpr Backend DefaultBackend = Backend::OmegaH;
#endif

  // Unstructured mesh — dispatches to MeshFields or native Omega_h backend
  [[nodiscard]] static LagrangeFieldFactory FromMesh(
    Omega_h::Mesh& mesh, int order, int num_components,
    CoordinateSystem coordinate_system,
    std::string global_id_name = "global",
    Backend backend = DefaultBackend);

  // Structured uniform grid — order-1 H1-conforming nodal field on a regular grid
  [[nodiscard]] static LagrangeFieldFactory FromUniformGrid(
    Rank1View<Real, HostMemorySpace> edge_length,
    Rank1View<Real, HostMemorySpace> bot_left,
    Rank1View<LO, HostMemorySpace>   divisions,
    int num_components,
    CoordinateSystem coordinate_system,
    int order = 1);

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout() const noexcept;

  [[nodiscard]] CoordinateSystem GetCoordinateSystem() const noexcept;

  [[nodiscard]] const FieldEvaluatorFactory<Real>& GetEvaluatorFactory() const noexcept;

  // Delegates PointEvaluator construction to the internal factory.
  [[nodiscard]] std::unique_ptr<PointEvaluator<Real>> CreatePointEvaluator(
    CoordinateView<HostMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const;

  // Returns a new FieldData<Real> pre-allocated for this layout.
  [[nodiscard]] std::unique_ptr<FieldData<Real>> CreateFieldData(
    FieldMetadata metadata = {}) const;

private:
  explicit LagrangeFieldFactory(
    std::shared_ptr<const FieldLayout> layout,
    std::function<std::unique_ptr<FieldData<Real>>(FieldMetadata)>
      create_field_data_fn,
    std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept;

  std::shared_ptr<const FieldLayout> layout_;
  std::function<std::unique_ptr<FieldData<Real>>(FieldMetadata)>
    create_field_data_fn_;
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory_;
};

} // namespace pcms

#endif // PCMS_LAGRANGE_FIELD_FACTORY_H
