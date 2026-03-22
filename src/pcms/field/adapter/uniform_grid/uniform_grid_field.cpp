#include "uniform_grid_field.h"
#include "uniform_grid_evaluator_factory.h"
#include "pcms/field/simple_field_data.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/assert.h"
#include <memory>

namespace pcms
{


template <unsigned Dim>
UniformGridField<Dim>::UniformGridField(
  std::shared_ptr<const UniformGridFieldLayout<Dim>> layout)
  : layout_(std::move(layout)),
    grid_(layout_->GetGrid()),
    dof_holder_data_("dof_holder_data", static_cast<size_t>(layout_->OwnedSize())),
    evaluator_factory_(std::make_shared<UniformGridEvaluatorFactory<Dim>>(layout_))
{
  PCMS_FUNCTION_TIMER;
  // Default to NEAREST_BOUNDARY for uniform grid fields
  out_of_bounds_mode_ = OutOfBoundsMode::NEAREST_BOUNDARY;
}

template <unsigned Dim>
Rank1View<const Real, HostMemorySpace> UniformGridField<Dim>::GetDOFHolderData()
  const
{
  PCMS_FUNCTION_TIMER;
  return make_const_array_view(dof_holder_data_);
}

template <unsigned Dim>
void UniformGridField<Dim>::SetDOFHolderData(
  Rank1View<const Real, HostMemorySpace> data)
{
  PCMS_FUNCTION_TIMER;
  PCMS_ALWAYS_ASSERT(data.size() == dof_holder_data_.size());

  for (size_t i = 0; i < data.size(); ++i) {
    dof_holder_data_[i] = data[i];
  }
}

template <unsigned Dim>
View<Dim, Real, HostMemorySpace> UniformGridField<Dim>::to_mdspan()
{
  PCMS_FUNCTION_TIMER;

  if constexpr (Dim == 2) {
    if (layout_->GetOrder() == 0) {
      return View<Dim, Real, HostMemorySpace>(
        dof_holder_data_.data(), grid_.divisions[0], grid_.divisions[1]);
    }
    return View<Dim, Real, HostMemorySpace>(
      dof_holder_data_.data(), grid_.divisions[0] + 1, grid_.divisions[1] + 1);
  } else if constexpr (Dim == 3) {
    if (layout_->GetOrder() == 0) {
      return View<Dim, Real, HostMemorySpace>(
        dof_holder_data_.data(), grid_.divisions[0], grid_.divisions[1],
        grid_.divisions[2]);
    }
    return View<Dim, Real, HostMemorySpace>(
      dof_holder_data_.data(), grid_.divisions[0] + 1, grid_.divisions[1] + 1,
      grid_.divisions[2] + 1);
  } else {
    static_assert(Dim == 2 || Dim == 3,
                  "to_mdspan only supports 2D or 3D uniform grids");
  }
}

template <unsigned Dim>
View<Dim, const Real, HostMemorySpace> UniformGridField<Dim>::to_mdspan() const
{
  PCMS_FUNCTION_TIMER;

  if constexpr (Dim == 2) {
    if (layout_->GetOrder() == 0) {
      return View<Dim, const Real, HostMemorySpace>(
        dof_holder_data_.data(), grid_.divisions[0], grid_.divisions[1]);
    }
    return View<Dim, const Real, HostMemorySpace>(
      dof_holder_data_.data(), grid_.divisions[0] + 1, grid_.divisions[1] + 1);
  } else if constexpr (Dim == 3) {
    if (layout_->GetOrder() == 0) {
      return View<Dim, const Real, HostMemorySpace>(
        dof_holder_data_.data(), grid_.divisions[0], grid_.divisions[1],
        grid_.divisions[2]);
    }
    return View<Dim, const Real, HostMemorySpace>(
      dof_holder_data_.data(), grid_.divisions[0] + 1, grid_.divisions[1] + 1,
      grid_.divisions[2] + 1);
  } else {
    static_assert(Dim == 2 || Dim == 3,
                  "to_mdspan only supports 2D or 3D uniform grids");
  }
}

template <unsigned Dim>
LocalizationHint UniformGridField<Dim>::GetLocalizationHint(
  CoordinateView<HostMemorySpace> coordinate_view) const
{
  PCMS_FUNCTION_TIMER;
  OutOfBoundsPolicy policy{this->out_of_bounds_mode_, this->fill_value_};
  auto ev = evaluator_factory_->CreatePointEvaluator(coordinate_view, policy);
  return LocalizationHint{std::shared_ptr<PointEvaluator<Real>>(std::move(ev))};
}

template <unsigned Dim>
void UniformGridField<Dim>::Evaluate(
  LocalizationHint location, FieldDataView<Real, HostMemorySpace> results) const
{
  PCMS_FUNCTION_TIMER;

  if (results.GetCoordinateSystem() !=
      layout_->GetDOFHolderCoordinates().GetCoordinateSystem()) {
    throw std::runtime_error("Coordinate system mismatch in Evaluate");
  }

  auto* ev = static_cast<PointEvaluator<Real>*>(location.data.get());

  // Create a temporary SimpleFieldData with the same layout so that the
  // UniformGridPointEvaluator layout-identity check passes. Then copy the
  // current DOF data into it before delegating.
  SimpleFieldData<Real> tmp(layout_, FieldMetadata{});
  tmp.SetDOFHolderDataHost(make_const_array_view(dof_holder_data_));

  auto values_1d = results.GetValues();
  LO n_pts = static_cast<LO>(values_1d.size());
  Rank2View<Real, HostMemorySpace> out(values_1d.data_handle(), n_pts, 1);
  ev->Evaluate(tmp, out);
}

template <unsigned Dim>
void UniformGridField<Dim>::EvaluateGradient(
  FieldDataView<Real, HostMemorySpace>)
{
  throw std::runtime_error("Not implemented");
}

template <unsigned Dim>
const FieldLayout& UniformGridField<Dim>::GetLayout() const
{
  return *layout_;
}

template <unsigned Dim>
bool UniformGridField<Dim>::CanEvaluateGradient()
{
  return false;
}

template <unsigned Dim>
LO UniformGridField<Dim>::CellIdToDofIndex(LO cell_id) const
{
  return cell_id;
}

// Explicit template instantiations
template class UniformGridField<2>;
template class UniformGridField<3>;

} // namespace pcms
