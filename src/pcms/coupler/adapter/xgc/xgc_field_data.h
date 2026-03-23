#ifndef PCMS_XGC_FIELD_DATA_H
#define PCMS_XGC_FIELD_DATA_H

#include "pcms/coupler/adapter/xgc/xgc_field_layout.h"
#include "pcms/field/field_data.h"
#include "pcms/utility/assert.h"
#include <mpi.h>
#include <memory>

namespace pcms
{

template <typename T>
class XGCFieldData : public FieldData<T>
{
public:
  XGCFieldData(std::shared_ptr<const XGCFieldLayout> layout,
               FieldMetadata metadata, Rank1View<T, HostMemorySpace> data,
               MPI_Comm plane_comm)
    : layout_(std::move(layout)),
      metadata_(metadata),
      data_(data),
      plane_comm_(plane_comm)
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

  MPI_Comm GetPlaneComm() const noexcept { return plane_comm_; }

private:
  std::shared_ptr<const XGCFieldLayout> layout_;
  FieldMetadata metadata_;
  Rank1View<T, HostMemorySpace> data_;
  MPI_Comm plane_comm_;
};

} // namespace pcms

#endif // PCMS_XGC_FIELD_DATA_H
