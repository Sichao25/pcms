#ifndef PCMS_COUPLING_ARRAYS_H
#define PCMS_COUPLING_ARRAYS_H
#include "mdspan/mdspan.hpp"
#include "pcms/utility/types.h"
#include "pcms/utility/memory_spaces.h"

#include <Omega_h_array.hpp>

namespace pcms
{

namespace detail
{

template <typename ElementType, typename MemorySpace>
struct memory_space_accessor : public Kokkos::default_accessor<ElementType>
{
  using memory_space = MemorySpace;

  // Default constructor
  memory_space_accessor() = default;

  // Converting constructor to allow adding const qualification
  template <typename OtherElementType,
            typename = std::enable_if_t<std::is_convertible_v<OtherElementType(*)[], ElementType(*)[]>>>
  constexpr memory_space_accessor(const memory_space_accessor<OtherElementType, MemorySpace>&) noexcept {}
};

} // namespace detail

template <typename ContainerType, typename ElementType, typename Extents,
          typename LayoutPolicy, typename AccessorPolicy>
auto make_mdspan(const ContainerType& /* unused */)
  -> Kokkos::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>
{
  static_assert(detail::dependent_always_false<ContainerType>::type,
                "creating mdspan is not implemented for type");
}

// TODO make_mdspan

template <int Rank, typename ElementType, typename MemorySpace,
          typename LayoutPolicy = Kokkos::layout_right,
          typename IndexType = LO>
using View = Kokkos::mdspan<
  ElementType, Kokkos::dextents<IndexType, Rank>, LayoutPolicy,
  detail::memory_space_accessor<std::remove_reference_t<ElementType>,
                                MemorySpace>>;

template <typename ElementType, typename MemorySpace, 
          typename LayoutPolicy = Kokkos::layout_right>
using Rank1View = View<1, ElementType, MemorySpace, LayoutPolicy>;

template <typename ElementType, typename MemorySpace,
          typename LayoutPolicy = Kokkos::layout_right>
using Rank2View = View<2, ElementType, MemorySpace, LayoutPolicy>;

template <typename ElementType, typename MemorySpace,
          typename LayoutPolicy = Kokkos::layout_right>
using Rank3View = View<3, ElementType, MemorySpace, LayoutPolicy>;

template <typename ElementType, typename MemorySpace,
          typename LayoutPolicy = Kokkos::layout_right>
using Rank4View = View<4, ElementType, MemorySpace, LayoutPolicy>;

template <typename MemorySpace>
using GlobalIDView = View<1, const GO, MemorySpace, Kokkos::layout_right, GO>;

namespace detail
{
template <typename T, typename = std::void_t<>>
struct HasValueType : std::false_type
{};
template <typename T>
struct HasValueType<T, std::void_t<typename T::value_type>> : std::true_type
{};

template <typename T, typename = std::void_t<>>
struct memory_space_selector
{
  using type = HostMemorySpace;
};
template <typename T>
struct memory_space_selector<T, std::void_t<typename T::memory_space>>
{
  using type = typename T::memory_space;
};

template <typename T>
using memory_space_selector_t = typename memory_space_selector<T>::type;

template <typename T, bool = false>
struct arr_trait;
template <typename T>
struct arr_trait<T, true>
{
  using type = typename T::value_type;
};

template <typename T, size_t N>
struct arr_trait<T[N]>
{
  using type = T;
};

template <typename T, size_t N>
struct arr_trait<std::array<T, N>>
{
  using type = T;
};

template <typename T>
using element_type_t = typename arr_trait<T, HasValueType<T>::value>::type;

// Layout selector: detects layout from source type
template <typename T, typename = std::void_t<>>
struct layout_selector
{
  using type = Kokkos::layout_right; // Default for host arrays and std::array
};

// Specialization for types with array_layout (Kokkos Views)
template <typename T>
struct layout_selector<T, std::void_t<typename T::array_layout>>
{
  using type = std::conditional_t<
    std::is_same_v<typename T::array_layout, Kokkos::LayoutLeft>,
    Kokkos::layout_left,
    Kokkos::layout_right>;
};

template <typename T>
using layout_selector_t = typename layout_selector<T>::type;

// Default layout selector for memory spaces
// GPU memory spaces typically use LayoutLeft for better coalescing
template <typename MemorySpace, typename = std::void_t<>>
struct default_layout_for_memory_space
{
  using type = Kokkos::layout_right; // Default for host
};

#ifdef PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE
// Specialization for device memory space - use LayoutLeft for GPU
template <>
struct default_layout_for_memory_space<DeviceMemorySpace>
{
  using type = Kokkos::layout_left;
};
#endif

template <typename MemorySpace>
using default_layout_for_memory_space_t = 
  typename default_layout_for_memory_space<MemorySpace>::type;

} // namespace detail
  // default implementation of make_array_view
template <typename T, typename MemorySpace = detail::memory_space_selector_t<T>,
          typename ElementType = detail::element_type_t<T>>
auto make_array_view(const T& array)
  -> Rank1View<const ElementType, MemorySpace>
{
  using std::data;
  using std::size;
  return Rank1View<const ElementType, MemorySpace>{data(array), size(array)};
}
template <typename T, typename MemorySpace = detail::memory_space_selector_t<T>,
          typename ElementType = detail::element_type_t<T>>
auto make_array_view(T& array) -> Rank1View<ElementType, MemorySpace>
{
  using std::data;
  using std::size;
  return Rank1View<ElementType, MemorySpace>{data(array), size(array)};
}

template <typename T, typename MemorySpace = detail::memory_space_selector_t<T>,
          typename ElementType = detail::element_type_t<T>>
auto make_const_array_view(T& array)
  -> Rank1View<const ElementType, MemorySpace>
{
  using std::data;
  using std::size;
  return Rank1View<const ElementType, MemorySpace>{data(array), size(array)};
}

// Helper functions to create mdspan views from Kokkos Views with layout detection
template <typename T, typename... Properties>
auto MakeRank1View(Kokkos::View<T*, Properties...>& view)
{
  using KokkosView = Kokkos::View<T*, Properties...>;
  using MemorySpace = typename KokkosView::memory_space;
  using LayoutPolicy = detail::layout_selector_t<KokkosView>;
  return Rank1View<T, MemorySpace, LayoutPolicy>(view.data(), view.extent(0));
}

template <typename T, typename... Properties>
auto MakeRank1View(const Kokkos::View<T*, Properties...>& view)
{
  using KokkosView = Kokkos::View<T*, Properties...>;
  using MemorySpace = typename KokkosView::memory_space;
  using LayoutPolicy = detail::layout_selector_t<KokkosView>;
  return Rank1View<const T, MemorySpace, LayoutPolicy>(view.data(), view.extent(0));
}

template <typename T, typename... Properties>
auto MakeRank2View(Kokkos::View<T**, Properties...>& view)
{
  using KokkosView = Kokkos::View<T**, Properties...>;
  using MemorySpace = typename KokkosView::memory_space;
  using LayoutPolicy = detail::layout_selector_t<KokkosView>;
  return Rank2View<T, MemorySpace, LayoutPolicy>(view.data(), view.extent(0), view.extent(1));
}

template <typename T, typename... Properties>
auto MakeRank2View(const Kokkos::View<T**, Properties...>& view)
{
  using KokkosView = Kokkos::View<T**, Properties...>;
  using MemorySpace = typename KokkosView::memory_space;
  using LayoutPolicy = detail::layout_selector_t<KokkosView>;
  return Rank2View<const T, MemorySpace, LayoutPolicy>(view.data(), view.extent(0), view.extent(1));
}

template <typename T, typename... Properties>
auto MakeConstRank2View(const Kokkos::View<T**, Properties...>& view)
{
  using KokkosView = Kokkos::View<T**, Properties...>;
  using MemorySpace = typename KokkosView::memory_space;
  using LayoutPolicy = detail::layout_selector_t<KokkosView>;
  return Rank2View<const T, MemorySpace, LayoutPolicy>(view.data(), view.extent(0), view.extent(1));
}

template <typename T>
auto MakeConstRank2View(Omega_h::Read<T> array, int dim)
{
  return Rank2View<const T, DeviceMemorySpace, Kokkos::layout_right>(array.data(), array.size() / dim, dim);
}

// utility function to deep copy between layout incompatible views
// layout imcompatible view can't be deep_copied directly between different
// memory spaces, the workaround is provided from
// https://kokkos.org/kokkos-core-wiki/API/core/view/deep_copy.html#how-to-get-layout-incompatible-views-copied
template <typename DestView, typename SrcView>
auto DeepCopyMismatchLayouts(DestView& dest, const SrcView& src)
{
  static_assert(Kokkos::is_view<DestView>::value &&
                  Kokkos::is_view<SrcView>::value,
                "Both arguments must be Kokkos::View types");

  using DestMemSpace = typename DestView::memory_space;
  using SrcMemSpace = typename SrcView::memory_space;

  if constexpr (std::is_same_v<DestMemSpace, HostMemorySpace> &&
                !std::is_same_v<SrcMemSpace, HostMemorySpace>) {
    // Device to Host: mirror the device view to host
    auto src_tmp = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), src);
    Kokkos::deep_copy(dest, src_tmp);
  } else if constexpr (!std::is_same_v<DestMemSpace, HostMemorySpace> &&
                       std::is_same_v<SrcMemSpace, HostMemorySpace>) {
    // Host to Device: create mirror on device from host view
    auto dest_tmp = Kokkos::create_mirror_view(dest);
    Kokkos::deep_copy(dest_tmp, src);
    Kokkos::deep_copy(dest, dest_tmp);
  } else {
    // Same memory space
    Kokkos::deep_copy(dest, src);
  }
}
template <typename DestView, typename SrcView>
void ConvertMismatchLayoutView2D(DestView& dest, const SrcView& src)
{
  Kokkos::parallel_for(
    "ConvertMismatchLayoutView2D", Kokkos::RangePolicy<typename DestView::execution_space>(0, src.extent(0)),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < src.extent(1); ++j) {
        dest(i, j) = src(i, j);
      }
    });
}

// utility function to fill a view with sequentially increasing values
template <typename T>
void iota_view(Kokkos::View<T*> view, T start = 0)
{
  Kokkos::parallel_for(
    "iota_view", view.extent(0), KOKKOS_LAMBDA(LO i) { view[i] = start + i; });
}

// utility function to copy device Rank1View to device Kokkos View
template <typename T>
void CopyDeviceRank1ViewToDeviceView(Kokkos::View<T*, DeviceMemorySpace> dest,
                                     Rank1View<const T, DeviceMemorySpace> src)
{
  if (dest.extent(0) != src.size()) {
    throw pcms_error("CopyDeviceRank1ViewToDeviceView: size mismatch");
  }
  Kokkos::parallel_for(
    "CopyDeviceRank1ViewToDeviceView", dest.extent(0),
    KOKKOS_LAMBDA(LO i) { dest(i) = src(i); });
}

// utility function to copy device Rank1View to host Kokkos View
template <typename T>
void CopyDeviceRank1ViewToHostView(Kokkos::View<T*, HostMemorySpace> dest,
                                   Rank1View<const T, DeviceMemorySpace> src)
{
  if (dest.extent(0) != src.size()) {
    throw pcms_error("CopyDeviceRank1ViewToHostView: size mismatch");
  }
  Kokkos::View<T*, DeviceMemorySpace> src_tmp("CopyDeviceRank1ViewToHostView_tmp", src.size());
  Kokkos::parallel_for(
    "CopyDeviceRank1ViewToHostView", src.size(),
    KOKKOS_LAMBDA(LO i) { src_tmp(i) = src(i); });
  Kokkos::deep_copy(dest, src_tmp);
}

template <typename T>
void CopyHostRank1ViewToDeviceView(Kokkos::View<T*, DeviceMemorySpace> dest,
                                    Rank1View<const T, HostMemorySpace> src)
{
  if (dest.extent(0) != src.size()) {
    throw pcms_error("CopyHostRank1ViewToDeviceView: size mismatch");
  }
  Kokkos::View<T*, HostMemorySpace> src_tmp("CopyHostRank1ViewToDeviceView_tmp", src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    src_tmp(i) = src(i);
  }
  Kokkos::deep_copy(dest, src_tmp);
}


// utility function to copy from Rank1View to Rank1View
template <typename T>
void CopyRank1ViewToHost(Rank1View<T, HostMemorySpace> dest,
                              Rank1View<const T, DeviceMemorySpace> src)
{
  if (dest.size() != src.size()) {
    throw pcms_error("CopyRank1ViewToHost: size mismatch");
  }
  Kokkos::View<T*, DeviceMemorySpace> src_tmp("CopyRank1ViewToHost_tmp", src.size());
  Kokkos::parallel_for(
    "CopyRank1ViewToHost", src.size(),
    KOKKOS_LAMBDA(LO i) { src_tmp(i) = src(i); });
  Kokkos::View<T*, HostMemorySpace> src_tmp_h("CopyRank1ViewToHost_tmp_h", src.size());
  Kokkos::deep_copy(src_tmp_h, src_tmp);
  for (size_t i = 0; i < src.size(); ++i) {
    dest(i) = src_tmp_h(i);
  }

}

template <typename T>
void CopyRank1ViewToDevice(Rank1View<T, DeviceMemorySpace> dest,
                                Rank1View<const T, HostMemorySpace> src)
{
  if (dest.size() != src.size()) {
    throw pcms_error("CopyRank1ViewToDevice: size mismatch");
  }
  Kokkos::View<T*, HostMemorySpace> src_tmp("CopyRank1ViewToDevice_tmp", src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    src_tmp(i) = src(i);
  }
  Kokkos::View<T*, DeviceMemorySpace> src_tmp_d("CopyRank1ViewToDevice_tmp_d", src.size());
  Kokkos::deep_copy(src_tmp_d, src_tmp);
  Kokkos::parallel_for("CopyRank1ViewToDevice", src.size(), KOKKOS_LAMBDA(LO i) {
    dest(i) = src_tmp_d(i);
  });
}


} // namespace pcms
#endif // PCMS_COUPLING_ARRAYS_H
