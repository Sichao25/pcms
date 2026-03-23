#ifndef FIELD_COMMUNICATOR2_H_
#define FIELD_COMMUNICATOR2_H_

#include "pcms/coupler/field_layout_communicator.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_data.h"
#include "pcms/coupler/field_serializer.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/inclusive_scan.h"
#include "pcms/utility/arrays.h"
#include <redev.h>
#include <memory>

namespace pcms
{

template <typename T>
class FieldCommunicator2
{
public:
  FieldCommunicator2(FieldLayoutCommunicator& layout_comm, FieldData<T>& field)
    : FieldCommunicator2(layout_comm, field, FieldSerializer<T>{})
  {
  }

  FieldCommunicator2(FieldLayoutCommunicator& layout_comm, FieldData<T>& field,
                     FieldSerializer<T> serializer)
    : comm_buffer_{},
      layout_comm_(layout_comm),
      field_(field),
      serializer_(std::move(serializer))
  {
    PCMS_ALWAYS_ASSERT(&layout_comm.GetLayout() == &field.GetLayout());
    comm_buffer_.resize(layout_comm.GetMsgSize());
    comm_ = layout_comm_.GetChannel().CreateComm<T>(layout_comm.GetName(),
                                                    layout_comm.GetMPIComm());
    layout_comm_.SetOutMessageLayout<T>(comm_);
  }

  void Send(redev::Mode mode = redev::Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    PCMS_ALWAYS_ASSERT(layout_comm_.GetChannel().InSendCommunicationPhase());
    auto buffer = make_array_view(comm_buffer_);
    serializer_.Serialize(field_, buffer, layout_comm_.GetPermutationArray());
    comm_.Send(buffer.data_handle(), mode);
  }

  void Receive()
  {
    PCMS_FUNCTION_TIMER;
    PCMS_ALWAYS_ASSERT(layout_comm_.GetChannel().InReceiveCommunicationPhase());
    // Current implementation requires that Receive is always called in Sync
    // mode because we make an immediate call to deserialize after a call to
    // receive.
    auto data = comm_.Recv(redev::Mode::Synchronous);
    serializer_.Deserialize(field_,
                            make_const_array_view(data),
                            layout_comm_.GetPermutationArray());
  }

private:
  std::vector<T> comm_buffer_;
  redev::BidirectionalComm<T> comm_;
  FieldLayoutCommunicator& layout_comm_;
  FieldData<T>& field_;
  FieldSerializer<T> serializer_;
};

template <typename T>
using FieldCommunicator2PtrT = std::unique_ptr<FieldCommunicator2<T>>;

using FieldCommunicator2Ptr =
  std::variant<FieldCommunicator2PtrT<int8_t>, FieldCommunicator2PtrT<int32_t>,
               FieldCommunicator2PtrT<int64_t>, FieldCommunicator2PtrT<float>,
               FieldCommunicator2PtrT<double>>;
} // namespace pcms

#endif // FIELD_COMMUNICATOR2_H_
