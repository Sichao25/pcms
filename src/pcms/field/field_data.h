#ifndef PCMS_FIELD_DATA_H
#define PCMS_FIELD_DATA_H

#include "field_metadata.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include <memory>
#include <variant>

namespace pcms
{

// FieldData<T> stores the data that parameterizes a field function, plus
// metadata. The interpretation of the stored data is defined entirely by the
// FieldEvaluatorFactory that consumes it:
//
//   - Mesh-backed fields:    DOF holder coefficients (nodal values, modal
//                            coefficients, etc.) at locations defined by the
//                            layout.
//   - Analytic functions:    function parameters (e.g. wavenumber, amplitude).
//   - ML surrogates:         model weights or a handle to model state.
//
// This generality means FieldData is not restricted to fields with spatial DOF
// locations. HasDOFHolderCoordinates() on the corresponding
// FieldEvaluatorFactory distinguishes spatially-located fields from those
// whose parameter data has no physical-space interpretation.
//
// Concrete backends manage the ownership and storage policy for the underlying
// data.
//
// Contract:
//   - GetMetadata() describes the stored coefficient data.
//   - The flattened DOF-holder data ordering must match the layout ordering.
//   - Views returned by GetDOFHolderDataHost()/GetDOFHolderDataDevice() remain
//     valid until this FieldData object is mutated or destroyed.
//   - SetDOFHolderDataHost()/SetDOFHolderDataDevice() replace the entire stored
//     coefficient array; partial updates are not part of this interface.
//   - Basic coefficient operations should be expressible from FieldData and
//     its associated FieldLayout; callers should not need the originating
//     factory for routine get/set workflows.
//
// NOTE: FieldData<T> is being introduced incrementally alongside the existing
// FieldT<T>. Concrete backends will migrate from FieldT<T> to FieldData<T>
// over time.
template <typename T>
class FieldData
{
public:
  using value_type = T;

  virtual const FieldMetadata& GetMetadata() const = 0;

  // Direct access to flattened DOFHolder-ordered coefficient data.
  // The returned view remains valid until the FieldData is mutated or
  // destroyed.
  virtual Rank1View<const T, HostMemorySpace> GetDOFHolderDataHost() const = 0;
  virtual void SetDOFHolderDataHost(
    Rank1View<const T, HostMemorySpace> values) = 0;

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  // The returned view remains valid until the FieldData is mutated or
  // destroyed.
  virtual Rank1View<const T, DeviceMemorySpace>
  GetDOFHolderDataDevice() const = 0;
  virtual void SetDOFHolderDataDevice(
    Rank1View<const T, DeviceMemorySpace> values) = 0;
#endif

  virtual ~FieldData() noexcept = default;
};

template <typename T>
using OwnedFieldDataPtrT = std::unique_ptr<FieldData<T>>;

using OwnedFieldDataPtr =
  std::variant<OwnedFieldDataPtrT<int8_t>, OwnedFieldDataPtrT<int32_t>,
               OwnedFieldDataPtrT<int64_t>, OwnedFieldDataPtrT<float>,
               OwnedFieldDataPtrT<double>>;

} // namespace pcms

#endif // PCMS_FIELD_DATA_H
