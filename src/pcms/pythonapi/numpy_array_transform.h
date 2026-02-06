#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Omega_h_array.hpp>
#include "pcms/arrays.h"
#include "pcms/types.h"


namespace py = pybind11;


namespace pcms {


// Helper function to convert numpy array to Rank1 View
template<typename T>
Rank1View<T, HostMemorySpace> numpy_to_view(py::array_t<T> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  T* data_ptr = reinterpret_cast<T*>(buf.ptr);
  Rank1View<T, HostMemorySpace> view(data_ptr, buf.shape[0]);
  return view;
}


// Helper function to convert numpy array to Kokkos View
template<typename T>
Kokkos::View<T*, HostMemorySpace> numpy_to_kokkos_view(py::array_t<T> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  Kokkos::View<T*, HostMemorySpace> view(
    reinterpret_cast<T*>(buf.ptr), buf.shape[0]);
  return view;
}


// Helper function to convert Rank1 View to array (creates a copy)
template<typename T>
py::array_t<typename std::remove_const<T>::type> view_to_numpy(Rank1View<T, HostMemorySpace> view) {
  // Create a copy to avoid lifetime issues and mdspan type registration
  using NonConstT = typename std::remove_const<T>::type;
  py::array_t<NonConstT> result(view.size());
  py::buffer_info buf = result.request();
  NonConstT* ptr = static_cast<NonConstT*>(buf.ptr);
  // Use element-wise copy to ensure proper mdspan access
  for (size_t i = 0; i < view.size(); ++i) {
    ptr[i] = view[i];
  }
  return result;
}


// Helper function to convert Kokkos View to array (creates a reference)
template<typename T>
py::array_t<T> kokkos_view_to_numpy(Kokkos::View<T*, HostMemorySpace> view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(view.extent(0))},  // shape
    {sizeof(T)},                                   // strides
    view.data(),                                   // data pointer
    py::cast(view)                                 // base object to manage lifetime
  );
}


// Helper to convert 2D numpy array to Kokkos View
template<typename T>
Kokkos::View<T**, HostMemorySpace> numpy_to_kokkos_view_2d(py::array_t<T> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be 2");
  }
  Kokkos::View<T**, HostMemorySpace> view(
    reinterpret_cast<T*>(buf.ptr), buf.shape[0], buf.shape[1]);
  return view;
}


// Helper to convert 2D Kokkos View to numpy array (creates a reference)
template<typename T>
py::array_t<T> kokkos_view_2d_to_numpy(Kokkos::View<T**, HostMemorySpace> view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(view.extent(0)), static_cast<py::ssize_t>(view.extent(1))},  // shape
    {sizeof(T) * view.extent(1), sizeof(T)},                                                 // strides (row-major)
    view.data(),                                                                             // data pointer
    py::cast(view)                                                                           // base object to manage lifetime
  );
}


// Helper to convert 1D numpy array to Omega_h::Read
template<typename T>
Omega_h::Read<T> numpy_to_omega_h_read(py::array_t<T> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  Kokkos::View<T*, Kokkos::DefaultExecutionSpace::memory_space,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>> view(
    reinterpret_cast<T*>(buf.ptr), buf.shape[0]);
  Omega_h::Write<T> write_view(view);
  Omega_h::Read<T> read_view(write_view);
  return read_view;
}


// Helper to convert Omega_h::Read to numpy array (creates a copy)
template<typename T>
py::array_t<T> omega_h_read_to_numpy(Omega_h::Read<T> read_view) {
  auto read_view_host = Omega_h::HostRead<T>(read_view);
  py::array_t<T> result(read_view_host.size());
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);
  for (Omega_h::LO i = 0; i < read_view_host.size(); ++i) {
    ptr[i] = read_view_host[i];
  }
  return result;
}


// Helper to convert 1D numpy array to Omega_h::Write
template<typename T>
Omega_h::Write<T> numpy_to_omega_h_write(py::array_t<T> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  // Get host mirror and copy data
  auto write_view_host = Omega_h::HostWrite<T>(buf.shape[0]);
  T* ptr = static_cast<T*>(buf.ptr);
  for (Omega_h::LO i = 0; i < buf.shape[0]; ++i) {
    write_view_host[i] = ptr[i];
  }
  auto write_view = Omega_h::Write<T>(write_view_host);
  return write_view;
}


// Helper to convert Omega_h::Write to numpy array (creates a reference)
template<typename T>
py::array_t<T> omega_h_write_to_numpy(Omega_h::Write<T> write_view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(write_view.size())},  // shape
    {sizeof(T)},                                     // strides
    write_view.data(),                               // data pointer
    py::cast(write_view)                             // base object to manage lifetime
  );
}


// Helper function to convert numpy array to Rank2 View
template<typename T>
Rank2View<T, HostMemorySpace> numpy_to_view_2d(py::array_t<T> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be 2");
  }
  Rank2View<T, HostMemorySpace> view(
    reinterpret_cast<T*>(buf.ptr), buf.shape[0], buf.shape[1]);
  return view;
}


// Helper function to convert Rank2 View to array (creates a reference)
template<typename T>
py::array_t<T> view_2d_to_numpy(Rank2View<T, HostMemorySpace> view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(view.extent(0)), static_cast<py::ssize_t>(view.extent(1))},  // shape
    {sizeof(T) * view.extent(1), sizeof(T)},                                                 // strides (row-major)
    view.data_handle(),                                                                      // data pointer
    py::cast(view)                                                                           // base object to manage lifetime
  );
}


// Helper function to convert Rank3 View to array (creates a reference)
template<typename T>
py::array_t<T> view_3d_to_numpy(Rank3View<T, HostMemorySpace> view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(view.extent(0)),
     static_cast<py::ssize_t>(view.extent(1)),
     static_cast<py::ssize_t>(view.extent(2))},  // shape
    {sizeof(T) * view.extent(1) * view.extent(2),
     sizeof(T) * view.extent(2),
     sizeof(T)},                                 // strides (row-major)
    view.data_handle(),                          // data pointer
    py::cast(view)                               // base object to manage lifetime
  );
}


// Helper function to convert mdspan View to array (creates a reference)
template<typename T>
py::array_t<typename std::remove_const<T>::type> mdspan_view_to_numpy(
  View<2, T, HostMemorySpace> view) {
  using NonConstT = typename std::remove_const<T>::type;
  py::array_t<NonConstT> result(
    {static_cast<py::ssize_t>(view.extent(0)),
     static_cast<py::ssize_t>(view.extent(1))});
  auto out = result.template mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < out.shape(0); ++i) {
    for (py::ssize_t j = 0; j < out.shape(1); ++j) {
      out(i, j) = view(i, j);
    }
  }
  return result;
}

template<typename T>
py::array_t<typename std::remove_const<T>::type> mdspan_view_to_numpy(
  View<3, T, HostMemorySpace> view) {
  using NonConstT = typename std::remove_const<T>::type;
  py::array_t<NonConstT> result(
    {static_cast<py::ssize_t>(view.extent(0)),
     static_cast<py::ssize_t>(view.extent(1)),
     static_cast<py::ssize_t>(view.extent(2))});
  auto out = result.template mutable_unchecked<3>();
  for (py::ssize_t i = 0; i < out.shape(0); ++i) {
    for (py::ssize_t j = 0; j < out.shape(1); ++j) {
      for (py::ssize_t k = 0; k < out.shape(2); ++k) {
        out(i, j, k) = view(i, j, k);
      }
    }
  }
  return result;
}


} // namespace pcms