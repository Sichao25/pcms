#ifndef PCMS_ADAPTER_OMEGA_H_LAGRANGE_FIELD_H
#define PCMS_ADAPTER_OMEGA_H_LAGRANGE_FIELD_H

// OmegaHLagrangeField delegates all localization and evaluation to
// OmegaHLagrangeEvaluatorFactory / OmegaHLagrangePointEvaluator. This
// eliminates the duplicated spatial search structure and evaluation logic that
// previously lived in both the field and the evaluator factory.

#include "omega_h_lagrange_common.h"
#include "omega_h_lagrange_evaluator_factory.h"
#include "omega_h_lagrange_layout.h"
#include "pcms/field/field.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/simple_field_data.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/profile.h"

#include <memory>

namespace pcms
{

template <typename T>
class OmegaHLagrangeField : public FieldT<T>
{
public:
  explicit OmegaHLagrangeField(
    std::shared_ptr<const OmegaHLagrangeLayout> layout)
    : layout_(std::move(layout)),
      evaluator_factory_(
        std::make_shared<OmegaHLagrangeEvaluatorFactory<T>>(layout_)),
      field_data_(std::make_shared<SimpleFieldData<T>>(layout_, FieldMetadata{}))
  {
  }

  LocalizationHint GetLocalizationHint(
    CoordinateView<HostMemorySpace> coordinate_view) const override
  {
    PCMS_FUNCTION_TIMER;
    OutOfBoundsPolicy policy{this->out_of_bounds_mode_, this->fill_value_};
    auto ev = evaluator_factory_->CreatePointEvaluator(coordinate_view, policy);
    return LocalizationHint{std::shared_ptr<PointEvaluator<T>>(std::move(ev))};
  }

  void Evaluate(LocalizationHint location,
                FieldDataView<T, HostMemorySpace> results) const override
  {
    PCMS_FUNCTION_TIMER;
    if (results.GetCoordinateSystem() !=
        layout_->GetDOFHolderCoordinates().GetCoordinateSystem()) {
      throw pcms_error("Coordinate system mismatch");
    }
    PCMS_ALWAYS_ASSERT(layout_->GetNumComponents() == 1 &&
                       "OmegaHLagrangeField old API only supports 1 component");
    auto* ev =
      static_cast<PointEvaluator<T>*>(location.data.get());
    auto values_1d = results.GetValues();
    LO n_pts = static_cast<LO>(values_1d.size());
    Rank2View<T, HostMemorySpace> out(values_1d.data_handle(), n_pts, 1);
    ev->Evaluate(*field_data_, out);
  }

  void EvaluateGradient(FieldDataView<T, HostMemorySpace> /*unused*/) override
  {
    throw pcms_error(
      "EvaluateGradient not implemented for OmegaHLagrangeField");
  }

  bool CanEvaluateGradient() override { return false; }

  Rank1View<const T, HostMemorySpace> GetDOFHolderData() const override
  {
    return field_data_->GetDOFHolderDataHost();
  }

  void SetDOFHolderData(Rank1View<const T, HostMemorySpace> data) override
  {
    field_data_->SetDOFHolderDataHost(data);
  }

  const FieldLayout& GetLayout() const override { return *layout_; }

  ~OmegaHLagrangeField() noexcept = default;

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  std::shared_ptr<OmegaHLagrangeEvaluatorFactory<T>> evaluator_factory_;
  std::shared_ptr<SimpleFieldData<T>> field_data_;
};

} // namespace pcms

#endif // PCMS_ADAPTER_OMEGA_H_LAGRANGE_FIELD_H
