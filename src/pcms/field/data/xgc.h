#ifndef PCMS_XGC_FIELD_DATA_H
#define PCMS_XGC_FIELD_DATA_H

#include "pcms/field/layout/xgc.h"
#include "pcms/field/field_data.h"
#include "pcms/utility/assert.h"
#include <memory>

namespace pcms
{

template <typename T>
class XGCFieldData : public FieldData<T>
{
public:
  XGCFieldData(std::shared_ptr<const XGCFieldLayout> layout,
               FieldMetadata metadata, Rank1View<T, HostMemorySpace> data)
    : layout_(std::move(layout)),
      metadata_(metadata),
      data_(data)
  {
    PCMS_ALWAYS_ASSERT(layout_ != nullptr);
    PCMS_ALWAYS_ASSERT(static_cast<LO>(data_.size()) ==
                       layout_->GetFullDataSize());
  }

  const FieldLayout& GetLayout() const override { return *layout_; }

  const FieldMetadata& GetMetadata() const override { return metadata_; }

  Rank1View<const T, HostMemorySpace> GetDOFHolderDataHost() const override
  {
    return Rank1View<const T, HostMemorySpace>(data_.data_handle(),
                                               data_.size());
  }

  void SetDOFHolderDataHost(Rank1View<const T, HostMemorySpace> values) override
  {
    if (values.size() != data_.size()) {
      throw pcms_error("XGCFieldData::SetDOFHolderDataHost: size mismatch");
    }
    for (size_t i = 0; i < values.size(); ++i) {
      data_(i) = values[i];
    }
  }

private:
  std::shared_ptr<const XGCFieldLayout> layout_;
  FieldMetadata metadata_;
  Rank1View<T, HostMemorySpace> data_;
};

} // namespace pcms

#endif // PCMS_XGC_FIELD_DATA_H
