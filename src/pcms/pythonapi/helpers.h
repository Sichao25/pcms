#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
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
  Rank1View<T, HostMemorySpace> view(
    reinterpret_cast<T*>(buf.ptr), buf.shape[0]);
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

// Helper function to convert Rank1 View to array
template<typename T>
py::array_t<T> view_to_numpy(Rank1View<T, HostMemorySpace> view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(view.size())},
    {sizeof(T)},
    view.data_handle(),
    py::cast(view)
  );
}

// Helper function to convert Kokkos View to array
template<typename T>
py::array_t<T> kokkos_view_to_numpy(Kokkos::View<T*, HostMemorySpace> view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(view.extent(0))},
    {sizeof(T)},
    view.data(),
    py::cast(view)
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

// Helper to convert 2D Kokkos View to numpy array
template<typename T>
py::array_t<T> kokkos_view_2d_to_numpy(Kokkos::View<T**, HostMemorySpace> view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(view.extent(0)), static_cast<py::ssize_t>(view.extent(1))},
    {sizeof(T) * view.extent(1), sizeof(T)},
    view.data(),
    py::cast(view)
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

// Helper to convert Omega_h::Read to numpy array
template<typename T>
py::array_t<T> omega_h_read_to_numpy(Omega_h::Read<T> read_view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(read_view.size())},
    {sizeof(T)},
    read_view.data_handle(),
    py::cast(read_view)
  );
}

// Helper to convert 1D numpy array to Omega_h::Write
template<typename T>
Omega_h::Write<T> numpy_to_omega_h_write(py::array_t<T> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  Kokkos::View<T*, HostMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> view(
    reinterpret_cast<T*>(buf.ptr), buf.shape[0]);
  Omega_h::Write<T> write_view(view);
  return write_view;
}

// Helper to convert Omega_h::Write to numpy array
template<typename T>
py::array_t<T> omega_h_write_to_numpy(Omega_h::Write<T> write_view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(write_view.size())},
    {sizeof(T)},
    write_view.data_handle(),
    py::cast(write_view)
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

// Helper function to convert Rank2 View to array
template<typename T>
py::array_t<T> view_2d_to_numpy(Rank2View<T, HostMemorySpace> view) {
  return py::array_t<T>(
    {static_cast<py::ssize_t>(view.extent(0)), static_cast<py::ssize_t>(view.extent(1))},
    {sizeof(T) * view.extent(1), sizeof(T)},
    view.data_handle(),
    py::cast(view)
  );
}

} // namespace pcms