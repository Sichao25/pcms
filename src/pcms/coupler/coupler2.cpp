#include "pcms/coupler/coupler2.h"

namespace pcms
{

FieldLayoutCommunicator& Application2::GetLayoutCommunicator(
  const FieldLayout& layout)
{
  PCMS_FUNCTION_TIMER;
  auto it = field_layout_communicators_.find(&layout);
  if (it != field_layout_communicators_.end()) {
    return *it->second;
  } else {
    std::cerr << "Field added with external layout\n";
    std::terminate();
  }
}

const FieldLayout& Application2::AddLayout(std::string name,
                                           std::shared_ptr<const FieldLayout> layout)
{
  return AddLayout(std::move(name), std::move(layout),
                   std::make_unique<GenericFieldExchangePlanner>());
}

const FieldLayout& Application2::AddLayout(
  std::string name, std::shared_ptr<const FieldLayout> layout,
  std::unique_ptr<FieldExchangePlanner> planner)
{
  layouts_.push_back(std::move(layout));
  const FieldLayout& layout_ref = *layouts_.back();
  field_layout_communicators_.emplace(
    &layout_ref, std::make_unique<FieldLayoutCommunicator>(
                   name, mpi_comm_, redev_, channel_, layout_ref,
                   std::move(planner)));
  return layout_ref;
}

void Application2::AddField(std::string name, OwnedFieldDataPtr field,
                            bool participates)
{
  PCMS_FUNCTION_TIMER;

  fields_.push_back(std::move(field));

  FieldCommunicator2Ptr field_communicator = std::visit(
    [this, &name](auto& field_data_ptr) -> FieldCommunicator2Ptr {
      using T =
        typename std::remove_reference_t<decltype(*field_data_ptr)>::value_type;
      FieldLayoutCommunicator& layout_communicator =
        GetLayoutCommunicator(field_data_ptr->GetLayout());
      return std::make_unique<FieldCommunicator2<T>>(layout_communicator,
                                                     *field_data_ptr);
    },
    fields_.back());

  auto [it, inserted] =
    field_communicators_.insert_or_assign(name, std::move(field_communicator));

  if (!inserted) {
    throw pcms_error("Field with this name already exists");
  }
}

} // namespace pcms
