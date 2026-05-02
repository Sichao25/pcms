#ifndef PCMS_FIELD_LAYOUT_H
#define PCMS_FIELD_LAYOUT_H
#include <map>
#include <memory>
#include <vector>
#include "pcms/discretization/discretization.h"
#include "pcms/utility/types.h"
#include "pcms/utility/arrays.h"
#include "coordinate_system.h"

namespace pcms
{

constexpr int ent_offsets_len = 5;
using EntOffsetsArray = std::array<size_t, ent_offsets_len>;

using ReversePartitionMap = std::map<pcms::LO, std::vector<pcms::LO>>;

class FieldLayout
{
public:
  virtual std::shared_ptr<const Discretization> GetDiscretization()
    const noexcept = 0;

  // number of components
  int virtual GetNumComponents() const = 0;

  // nodes for standard lagrange FEM
  LO virtual GetNumOwnedDofHolder() const = 0;
  GO virtual GetNumGlobalDofHolder() const = 0;

  // size of buffer that needs to be allocated to represent the field
  // # components * NumDOFHolder
  LO OwnedSize() const { return GetNumComponents() * GetNumOwnedDofHolder(); };
  GO GlobalSize() const
  {
    return GetNumComponents() * GetNumGlobalDofHolder();
  };

  virtual Rank1View<const bool, HostMemorySpace> GetOwnedHost() const = 0;
  virtual GlobalIDView<HostMemorySpace> GetGidsHost() const = 0;

  // returns true if the field layout is distributed
  // if the field layout is distributed, the owned and global dofs are the same
  [[nodiscard]] virtual bool IsDistributed() const = 0;

  // This class should construct the permutation arrays that are needed
  // for serialization / deserialization
  //

  virtual EntOffsetsArray GetEntOffsets() const = 0;

  virtual CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates() const = 0;

  virtual int GetDimension() const = 0;

  virtual Rank1View<const LO, HostMemorySpace>
  GetDOFHolderClassificationDimensionsHost() const = 0;

  virtual Rank1View<const LO, HostMemorySpace>
  GetDOFHolderClassificationIdsHost() const = 0;

  virtual ~FieldLayout() noexcept = default;
};

} // namespace pcms
#endif // PCMS_FIELD_LAYOUT_H
