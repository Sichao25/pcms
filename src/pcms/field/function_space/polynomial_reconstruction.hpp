#ifndef PCMS_POLYNOMIAL_RECONSTRUCTION_FUNCTION_SPACE_H
#define PCMS_POLYNOMIAL_RECONSTRUCTION_FUNCTION_SPACE_H

#include "pcms/field/field.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/function_space.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"

#include <Omega_h_mesh.hpp>
#include <memory>

namespace pcms
{

template <typename T>
class FieldEvaluatorFactory;

class PolynomialReconstructionFunctionSpace : public FunctionSpace
{
public:
  [[nodiscard]] static PolynomialReconstructionFunctionSpace Create(
    Rank2View<Real, HostMemorySpace> coords, CoordinateSystem coordinate_system,
    MLSOptions options = {});

  [[nodiscard]] static PolynomialReconstructionFunctionSpace FromMesh(
    Omega_h::Mesh& mesh, int source_entity_dim,
    CoordinateSystem coordinate_system, MLSOptions options = {});

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout()
    const noexcept override;

  [[nodiscard]] CoordinateSystem GetCoordinateSystem() const noexcept override;

protected:
  [[nodiscard]] FieldVariant CreateFieldImpl(
    Type value_type, FieldMetadata metadata) const override;

  [[nodiscard]] FieldVariant CreateFieldImpl(
    FieldDataVariant data) const override;

  [[nodiscard]] PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, const EvaluationRequest& request) const override;

private:
  explicit PolynomialReconstructionFunctionSpace(
    std::shared_ptr<const FieldLayout> layout,
    std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept;

  std::shared_ptr<const FieldLayout> layout_;
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory_;
};

} // namespace pcms

#endif // PCMS_POLYNOMIAL_RECONSTRUCTION_FUNCTION_SPACE_H
