#ifndef FIELD_LAYOUT_COMMUNICATOR_H_
#define FIELD_LAYOUT_COMMUNICATOR_H_

#include "pcms/field/field_layout.h"
#include "field_exchange_planner.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/arrays.h"
#include <redev.h>
#include <memory>

namespace pcms
{

class FieldLayoutCommunicator
{
public:
  FieldLayoutCommunicator(const std::string& name, MPI_Comm mpi_comm,
                          redev::Redev& redev, redev::Channel& channel,
                          const FieldLayout& layout);

  FieldLayoutCommunicator(const std::string& name, MPI_Comm mpi_comm,
                          redev::Redev& redev, redev::Channel& channel,
                          const FieldLayout& layout,
                          std::unique_ptr<FieldExchangePlanner> planner);

  Rank1View<const pcms::LO, pcms::HostMemorySpace> GetPermutationArray() const;

  const std::string& GetName() const;

  const FieldLayout& GetLayout() const;

  size_t GetMsgSize() const;

  redev::Channel& GetChannel() const;

  MPI_Comm& GetMPIComm();

  template <typename T>
  void SetOutMessageLayout(redev::BidirectionalComm<T>& comm)
  {
    comm.SetOutMessageLayout(plan_.dest_ranks, plan_.offsets);
  }

  void UpdateLayout();

  void UpdateLayoutNull();

private:
  MPI_Comm mpi_comm_;
  redev::Channel& channel_;
  redev::BidirectionalComm<GO> gid_comm_;
  ExchangePlan plan_;
  const FieldLayout& layout_;
  std::string name_;
  redev::Redev& redev_;
  std::unique_ptr<FieldExchangePlanner> planner_;
};
} // namespace pcms

#endif // FIELD_LAYOUT_COMMUNICATOR_H_
