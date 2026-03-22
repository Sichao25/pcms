#ifndef PCMS_UNIFORM_GRID_FIELD_H
#define PCMS_UNIFORM_GRID_FIELD_H

#include "pcms/field/adapter/uniform_grid/uniform_grid_field_layout.h"
#include "pcms/utility/types.h"
#include "pcms/field/field.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/utility/uniform_grid.h"
#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

// Forward declaration to avoid circular include with uniform_grid_evaluator_factory.h
// (which includes this header for UniformGridFieldLocalizationHint).
template <unsigned Dim>
class UniformGridEvaluatorFactory;

// Localization hint for UniformGrid — stores per-point cell indices,
// coordinates, and out-of-bounds state produced by GetLocalizationHint or
// UniformGridEvaluatorFactory::CreatePointEvaluator.
// This localization hint will be removed when
template <unsigned Dim>
struct UniformGridFieldLocalizationHint
{
  UniformGridFieldLocalizationHint(
    Kokkos::View<LO*, HostMemorySpace> cell_indices,
    Kokkos::View<Real**, HostMemorySpace> coordinates,
    OutOfBoundsMode mode,
    Kokkos::View<bool*, HostMemorySpace> is_out_of_bounds,
    size_t num_out_of_bounds)
    : cell_indices_(cell_indices),
      coordinates_(coordinates),
      mode_(mode),
      is_out_of_bounds_(is_out_of_bounds),
      num_out_of_bounds_(num_out_of_bounds)
  {
  }

  Kokkos::View<LO*, HostMemorySpace>   cell_indices_;
  Kokkos::View<Real**, HostMemorySpace> coordinates_;
  OutOfBoundsMode                       mode_;
  Kokkos::View<bool*, HostMemorySpace>  is_out_of_bounds_;
  size_t                                num_out_of_bounds_;
};

template <unsigned Dim = 2>
class UniformGridField : public FieldT<Real>
{
public:
  UniformGridField(std::shared_ptr<const UniformGridFieldLayout<Dim>> layout);

  LocalizationHint GetLocalizationHint(
    CoordinateView<HostMemorySpace> coordinate_view) const override;

  void Evaluate(LocalizationHint location,
                FieldDataView<Real, HostMemorySpace> results) const override;

  void EvaluateGradient(FieldDataView<Real, HostMemorySpace> results) override;

  const FieldLayout& GetLayout() const override;

  bool CanEvaluateGradient() override;

  Rank1View<const Real, HostMemorySpace> GetDOFHolderData() const override;
  void SetDOFHolderData(Rank1View<const Real, HostMemorySpace> data) override;

  View<Dim, Real, HostMemorySpace> to_mdspan();
  View<Dim, const Real, HostMemorySpace> to_mdspan() const;

  ~UniformGridField() noexcept = default;

private:
  LO CellIdToDofIndex(LO cell_id) const;

  std::shared_ptr<const UniformGridFieldLayout<Dim>> layout_;
  const UniformGrid<Dim>& grid_;
  Kokkos::View<Real*, HostMemorySpace> dof_holder_data_;
  // Evaluator factory used to delegate GetLocalizationHint and Evaluate.
  // Stored as shared_ptr because UniformGridEvaluatorFactory has a reference
  // member and is therefore not movable/copyable.
  std::shared_ptr<UniformGridEvaluatorFactory<Dim>> evaluator_factory_;
};

using UniformGridField2D = UniformGridField<2>;

} // namespace pcms

#endif // PCMS_UNIFORM_GRID_FIELD_H
