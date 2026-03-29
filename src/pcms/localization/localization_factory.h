#ifndef PCMS_LOCALIZATION_LOCALIZATION_FACTORY_H
#define PCMS_LOCALIZATION_LOCALIZATION_FACTORY_H

#include "pcms/localization/adj_search.hpp"
#include "pcms/field/coordinate_system.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/discretization/discretization.h"

namespace pcms
{

// Abstract interface for building MLS support structures.
//
// A LocalizationFactory encapsulates the source geometry and search strategy
// (N^2 point-cloud search or mesh-adjacency BFS). Concrete implementations
// are created at FunctionSpace construction time and reused across repeated
// CreatePointEvaluator calls with different target coordinate sets.
class LocalizationFactory
{
public:
  virtual ~LocalizationFactory() = default;

  virtual SupportResults Build(
    CoordinateView<HostMemorySpace> target_coords) const = 0;

  virtual SupportResults Build(CoordinateView<HostMemorySpace> target_coords,
                               const Discretization& /* target_disc */) const
  {
    return Build(target_coords);
  }
};

} // namespace pcms

#endif // PCMS_LOCALIZATION_LOCALIZATION_FACTORY_H
