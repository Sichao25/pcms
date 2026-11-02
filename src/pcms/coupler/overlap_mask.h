#ifndef PCMS_OVERLAP_MASK_H
#define PCMS_OVERLAP_MASK_H

#include "pcms/field/field_layout.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include <Omega_h_array.hpp>
#include <functional>

namespace pcms
{

struct OverlapMask
{
  Kokkos::View<bool*, HostMemorySpace> is_overlap_;
  std::function<int8_t(int, int)> in_overlap_func_;

  // Default constructor: all DOF holders are in overlap
  OverlapMask(size_t size) : is_overlap_("overlap_info", size)
  {
    Kokkos::deep_copy(is_overlap_, true);
  }

  // Construct from a function that determines overlap from a DOF holder's
  // classification (dimension, id). Holders without classification are passed
  // as (-1, -1).
  explicit OverlapMask(std::function<int8_t(int, int)> in_overlap)
    : in_overlap_func_(std::move(in_overlap))
  {
  }

  // Construct from a precomputed per-DOF-holder host mask (e.g. an MFEM
  // subdomain selected by element attribute). Indexed by local DOF holder.
  OverlapMask(size_t size, Kokkos::View<bool*, HostMemorySpace> is_overlap_host)
    : is_overlap_("overlap_info", size)
  {
    PCMS_ALWAYS_ASSERT(is_overlap_host.extent(0) == size);
    Kokkos::deep_copy(is_overlap_, is_overlap_host);
  }

  // Construct from Omega_h host array
  OverlapMask(size_t size, Omega_h::HostRead<Omega_h::I8> is_overlap_host)
    : is_overlap_("overlap_info", size)
  {
    for (size_t i = 0; i < size; ++i) {
      is_overlap_[i] = static_cast<bool>(is_overlap_host[i]);
    }
  }

  // Construct from Omega_h device array
  OverlapMask(size_t size, Omega_h::Read<Omega_h::I8> is_overlap_device)
    : is_overlap_("overlap_info", size)
  {
    auto is_overlap_host = Omega_h::HostRead(is_overlap_device);
    for (size_t i = 0; i < size; ++i) {
      is_overlap_[i] = static_cast<bool>(is_overlap_host[i]);
    }
  }

  // Get the mask, evaluating the function against the layout if one was
  // given. The exchange plan is owned-indexed, so the result has one entry per
  // owned DOF holder. A mask stored per local DOF holder (e.g. from a
  // classification function or an MFEM attribute) is gathered to the owned
  // holders; one stored per owned holder is returned as is.
  Kokkos::View<const bool*, HostMemorySpace> GetMask(
    const FieldLayout& layout) const
  {
    Kokkos::View<const bool*, HostMemorySpace> mask = is_overlap_;
    if (in_overlap_func_) {
      auto class_dims = layout.GetDOFHolderClassificationDimensionsHost();
      auto class_ids = layout.GetDOFHolderClassificationIdsHost();
      Kokkos::View<bool*, HostMemorySpace> is_overlap("overlap_info",
                                                      class_dims.size());
      for (size_t i = 0; i < is_overlap.extent(0); ++i) {
        is_overlap[i] =
          static_cast<bool>(in_overlap_func_(class_dims[i], class_ids[i]));
      }
      mask = is_overlap;
    }
    const auto num_owned = static_cast<size_t>(layout.GetNumOwnedDofHolder());
    if (layout.OwnedIsLocal() || mask.extent(0) == num_owned) {
      return mask;
    }
    PCMS_ALWAYS_ASSERT(mask.extent(0) ==
                       static_cast<size_t>(layout.GetNumLocalDofHolder()));
    const auto owned_to_local = layout.GetOwnedToLocalHost();
    Kokkos::View<bool*, HostMemorySpace> owned_mask("owned_overlap_info",
                                                    num_owned);
    for (size_t o = 0; o < num_owned; ++o) {
      owned_mask[o] = mask[owned_to_local(o)];
    }
    return owned_mask;
  }
};

} // namespace pcms

#endif // PCMS_OVERLAP_MASK_H
