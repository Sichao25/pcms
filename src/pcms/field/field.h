#ifndef PCMS_COUPLING_FIELD_H
#define PCMS_COUPLING_FIELD_H

#include "field_data.h"
#include "field_evaluator_factory.h"
#include "field_layout.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <memory>

namespace pcms
{

// Field<T> is a composed per-field object: it owns coefficient data and holds
// a shared reference to the evaluator factory so the function space stays alive
// as long as the field does.
//
// Fields are typically created via LagrangeFunctionSpace::CreateField().
// They are move-only (unique_ptr member).
template <typename T>
class Field
{
public:
  Field(std::shared_ptr<const FieldEvaluatorFactory<Real>> evaluator_factory,
        std::unique_ptr<FieldData<T>> data)
    : evaluator_factory_(std::move(evaluator_factory)),
      data_(std::move(data))
  {
  }

  Field(Field&&) = default;
  Field& operator=(Field&&) = default;
  Field(const Field&)            = delete;
  Field& operator=(const Field&) = delete;

  FieldData<T>& GetData() noexcept { return *data_; }
  const FieldData<T>& GetData() const noexcept { return *data_; }

  const FieldLayout& GetLayout() const { return data_->GetLayout(); }

  // Returns the evaluator factory for this field's function space.
  // Null if the field was not created with an evaluatable function space.
  const FieldEvaluatorFactory<Real>* GetEvaluatorFactory() const noexcept
  {
    return evaluator_factory_.get();
  }

  Rank1View<const T, HostMemorySpace> GetDOFHolderDataHost() const
  {
    return data_->GetDOFHolderDataHost();
  }

  void SetDOFHolderDataHost(Rank1View<const T, HostMemorySpace> v)
  {
    data_->SetDOFHolderDataHost(v);
  }

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  Rank1View<const T, DeviceMemorySpace> GetDOFHolderDataDevice() const
  {
    return data_->GetDOFHolderDataDevice();
  }

  void SetDOFHolderDataDevice(Rank1View<const T, DeviceMemorySpace> v)
  {
    data_->SetDOFHolderDataDevice(v);
  }
#endif

private:
  std::shared_ptr<const FieldEvaluatorFactory<Real>> evaluator_factory_;
  std::unique_ptr<FieldData<T>> data_;
};

} // namespace pcms

#endif // PCMS_COUPLING_FIELD_H
