#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_ADAPTER2_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_ADAPTER2_H

#include "mesh_fields_evaluator_factory.h"
#include "mesh_fields_field_data.h"
#include "pcms/field/field.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_metadata.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/assert.h"

namespace pcms
{

// TODO template over possible MeshFieldsAdapter2Types
template <typename T>
class MeshFieldsAdapter2 : public FieldT<T>
{
public:
  explicit MeshFieldsAdapter2(
    std::shared_ptr<const MeshFieldsAdapterLayout> layout);

  LocalizationHint GetLocalizationHint(
    CoordinateView<HostMemorySpace> coordinate_view) const override;

  void Evaluate(LocalizationHint location,
                FieldDataView<T, HostMemorySpace> results) const override;

  void EvaluateGradient(FieldDataView<T, HostMemorySpace> results) override;

  const FieldLayout& GetLayout() const override;

  bool CanEvaluateGradient() override;

  Rank1View<const T, HostMemorySpace> GetDOFHolderData() const override;
  void SetDOFHolderData(Rank1View<const T, HostMemorySpace> data) override;

  ~MeshFieldsAdapter2() noexcept = default;

private:
  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  std::shared_ptr<MeshFieldsEvaluatorFactory<T>> evaluator_factory_;
  std::shared_ptr<MeshFieldsFieldData<T>> field_data_;
};

/*
 * MeshFieldsAdapter2 Implementation
 */
template <typename T>
inline MeshFieldsAdapter2<T>::MeshFieldsAdapter2(
  std::shared_ptr<const MeshFieldsAdapterLayout> layout)
  : layout_(std::move(layout)),
    evaluator_factory_(
      std::make_shared<MeshFieldsEvaluatorFactory<T>>(layout_)),
    field_data_(
      std::make_shared<MeshFieldsFieldData<T>>(layout_, FieldMetadata{}))
{
}

template <typename T>
inline Rank1View<const T, HostMemorySpace>
MeshFieldsAdapter2<T>::GetDOFHolderData() const
{
  PCMS_FUNCTION_TIMER;
  return field_data_->GetDOFHolderDataHost();
}

template <typename T>
inline void MeshFieldsAdapter2<T>::SetDOFHolderData(
  Rank1View<const T, HostMemorySpace> data)
{
  PCMS_FUNCTION_TIMER;
  field_data_->SetDOFHolderDataHost(data);
}

template <typename T>
inline LocalizationHint MeshFieldsAdapter2<T>::GetLocalizationHint(
  CoordinateView<HostMemorySpace> coordinate_view) const
{
  PCMS_FUNCTION_TIMER;
  OutOfBoundsPolicy policy{this->out_of_bounds_mode_, this->fill_value_};
  auto ev = evaluator_factory_->CreatePointEvaluator(coordinate_view, policy);
  return LocalizationHint{std::shared_ptr<PointEvaluator<T>>(std::move(ev))};
}

template <typename T>
inline void MeshFieldsAdapter2<T>::Evaluate(
  LocalizationHint location, FieldDataView<T, HostMemorySpace> results) const
{
  PCMS_FUNCTION_TIMER;
  if (results.GetCoordinateSystem() !=
      layout_->GetDOFHolderCoordinates().GetCoordinateSystem()) {
    throw pcms_error("Coordinate system mismatch");
  }
  PCMS_ALWAYS_ASSERT(layout_->GetNumComponents() == 1 &&
                     "MeshFieldsAdapter2 old API only supports 1 component");
  auto* ev = static_cast<PointEvaluator<T>*>(location.data.get());
  auto values_1d = results.GetValues();
  LO n_pts = static_cast<LO>(values_1d.size());
  Rank2View<T, HostMemorySpace> out(values_1d.data_handle(), n_pts, 1);
  ev->Evaluate(*field_data_, out);
}

template <typename T>
inline void MeshFieldsAdapter2<T>::EvaluateGradient(
  FieldDataView<T, HostMemorySpace> /* unused */)
{
  throw pcms_error("EvaluateGradient not implemented for MeshFieldsAdapter2");
}

template <typename T>
inline const FieldLayout& MeshFieldsAdapter2<T>::GetLayout() const
{
  return *layout_;
}

template <typename T>
inline bool MeshFieldsAdapter2<T>::CanEvaluateGradient()
{
  return false;
}

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_ADAPTER2_H
