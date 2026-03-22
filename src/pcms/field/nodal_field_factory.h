#ifndef PCMS_NODAL_FIELD_FACTORY_H
#define PCMS_NODAL_FIELD_FACTORY_H

#include "field.h"
#include "pcms/field/field_layout.h"
#include "coordinate_system.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"

#include <functional>
#include <memory>

namespace pcms
{

class NodalFieldFactory
{
public:
  [[nodiscard]] static NodalFieldFactory Create(
    Rank2View<Real, HostMemorySpace> coords,
    CoordinateSystem coordinate_system);

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout() const noexcept;
  [[nodiscard]] std::unique_ptr<FieldT<Real>> CreateFieldReal() const;

private:
  explicit NodalFieldFactory(
    std::shared_ptr<const FieldLayout> layout,
    std::function<std::unique_ptr<FieldT<Real>>()> create_fn) noexcept;

  std::shared_ptr<const FieldLayout> layout_;
  std::function<std::unique_ptr<FieldT<Real>>()> create_fn_;
};

} // namespace pcms

#endif // PCMS_NODAL_FIELD_FACTORY_H
