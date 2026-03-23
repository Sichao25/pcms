#ifndef PCMS_TEST_FIELD_TEST_UTILS_H
#define PCMS_TEST_FIELD_TEST_UTILS_H

#include <catch2/catch_approx.hpp>
#include <Kokkos_Core.hpp>
#include "pcms/field/field_data.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/coupler/field_serializer.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include <cmath>
#include <vector>

// Shared utilities for field tests that apply equally to MeshFields-backed
// and native OmegaH-backed field implementations.

namespace pcms::test
{

// Affine test function — exactly representable on linear elements.
KOKKOS_INLINE_FUNCTION Real linear_f(Real x, Real y)
{
  return x + 2.0 * y;
}

// Interior test points for a unit [0,1]^2 box mesh.
inline std::vector<Real> StandardEvalCoords2D()
{
  return {0.1, 0.2, 0.5, 0.5, 0.7, 0.3, 0.9, 0.1, 0.2, 0.8};
}

template <typename ExecutionSpace, typename Func>
inline std::vector<Real> EvaluateReferenceFunction(
  const std::vector<Real>& pts, Func func)
{
  using MemorySpace = typename ExecutionSpace::memory_space;

  int n = static_cast<int>(pts.size()) / 2;
  Kokkos::View<Real *[2], MemorySpace> pts_device("pts_device", n);
  auto pts_host =
    Kokkos::View<const Real **, HostMemorySpace>(pts.data(), n, 2);
  deep_copy_mismatch_layouts(pts_device, pts_host);

  Kokkos::View<Real *, MemorySpace> expected_device("expected_device", n);
  Kokkos::parallel_for(
    "field_test_utils_expected", Kokkos::RangePolicy<ExecutionSpace>(0, n),
    KOKKOS_LAMBDA(int i) {
      expected_device(i) = func(pts_device(i, 0), pts_device(i, 1));
    });

  auto expected_host =
    Kokkos::create_mirror_view_and_copy(HostMemorySpace(), expected_device);
  return std::vector<Real>(expected_host.data(),
                           expected_host.data() + expected_host.extent(0));
}

// Set scalar DOF data by sampling func at each DOF-holder coordinate.
template <typename ExecutionSpace = DefaultExecutionSpace, typename Func>
inline void SetField(FieldData<Real>& field, Func func)
{
  using MemorySpace = typename ExecutionSpace::memory_space;

  auto dof_coords = field.GetDOFHolderCoordinatesHost().GetCoordinates();
  int n = static_cast<int>(dof_coords.extent(0));
  Kokkos::View<Real *[2], MemorySpace> coords_device("coords_device", n);
  auto coords_host = Kokkos::View<const Real **, HostMemorySpace>(
    dof_coords.data_handle(), n, 2);
  deep_copy_mismatch_layouts(coords_device, coords_host);

  Kokkos::View<Real *, MemorySpace> data_device("data_device", n);
  Kokkos::parallel_for(
    "field_test_utils_set_field", Kokkos::RangePolicy<ExecutionSpace>(0, n),
    KOKKOS_LAMBDA(int i) {
      data_device(i) = func(coords_device(i, 0), coords_device(i, 1));
    });

  auto data_host = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), data_device);
  field.SetDOFHolderDataHost(
    Rank1View<const Real, HostMemorySpace>(data_host.data(), n));
}

// Check that serialize followed by deserialize round-trips the data.
// Uses an identity permutation so permutation[i] = i.
inline void CheckSerializeDeserialize(FieldData<Real>& field)
{
  auto data_before = field.GetDOFHolderDataHost();
  int n = static_cast<int>(data_before.size());

  std::vector<Real> buffer(n);
  std::vector<LO> perm(n);
  for (int i = 0; i < n; ++i)
    perm[i] = i;

  Rank1View<Real, HostMemorySpace> buf_view(buffer.data(), n);
  Rank1View<const LO, HostMemorySpace> perm_view(perm.data(), n);

  FieldSerializer<Real> serializer;
  serializer.Serialize(field, buf_view, perm_view);
  serializer.Deserialize(field, Rank1View<const Real, HostMemorySpace>(buf_view),
                         perm_view);

  auto data_after = field.GetDOFHolderDataHost();
  REQUIRE(data_after.size() == data_before.size());
  for (int i = 0; i < n; ++i) {
    REQUIRE(data_after[i] == Catch::Approx(data_before[i]));
  }
}

// Evaluate field at explicit test points using a PointEvaluator and check results.
template <typename ExecutionSpace = DefaultExecutionSpace, typename Func>
void CheckEvaluation(const PointEvaluator<Real>& evaluator,
                     FieldData<Real>& field_data,
                     const std::vector<Real>& pts, Func func,
                     double abs_tol = 1e-10)
{
  int n = static_cast<int>(pts.size()) / 2;
  std::vector<Real> eval(n);
  Rank2View<Real, HostMemorySpace> out(eval.data(), n, 1);
  evaluator.Evaluate(field_data, out);

  auto expected = EvaluateReferenceFunction<ExecutionSpace>(pts, func);

  for (int i = 0; i < n; ++i) {
    INFO("Point " << i << " (" << pts[2*i] << ", " << pts[2*i+1] << ")"
         << " got=" << eval[i] << " expected=" << expected[i]);
    REQUIRE(eval[i] == Catch::Approx(expected[i]).margin(abs_tol));
  }
}

// Overload that creates the evaluator from any factory with CreatePointEvaluator
// and GetCoordinateSystem (e.g. LagrangeFieldFactory, FieldEvaluatorFactory<Real>).
template <typename Factory, typename ExecutionSpace = DefaultExecutionSpace,
          typename Func>
void CheckEvaluation(const Factory& factory,
                     FieldData<Real>& field_data,
                     const std::vector<Real>& pts, Func func,
                     double abs_tol = 1e-10)
{
  int n = static_cast<int>(pts.size()) / 2;
  Rank2View<const Real, HostMemorySpace> coords_view(pts.data(), n, 2);
  CoordinateView<HostMemorySpace> cv{factory.GetCoordinateSystem(), coords_view};
  auto evaluator = factory.CreatePointEvaluator(cv);
  CheckEvaluation<ExecutionSpace>(*evaluator, field_data, pts, func, abs_tol);
}

// Evaluate field at points known to be outside the mesh and verify fill value.
inline void CheckFillMode(const PointEvaluator<Real>& evaluator,
                          FieldData<Real>& field_data, Real fill_value,
                          const std::vector<Real>& outside_pts)
{
  int n = static_cast<int>(outside_pts.size()) / 2;
  std::vector<Real> eval(n);
  Rank2View<Real, HostMemorySpace> out(eval.data(), n, 1);
  evaluator.Evaluate(field_data, out);

  for (int i = 0; i < n; ++i) {
    REQUIRE(eval[i] == fill_value);
  }
}

// Overload that creates the evaluator from any factory with FILL policy.
template <typename Factory>
void CheckFillMode(const Factory& factory,
                   FieldData<Real>& field_data, Real fill_value,
                   const std::vector<Real>& outside_pts)
{
  int n = static_cast<int>(outside_pts.size()) / 2;
  Rank2View<const Real, HostMemorySpace> coords_view(outside_pts.data(), n, 2);
  CoordinateView<HostMemorySpace> cv{factory.GetCoordinateSystem(), coords_view};
  OutOfBoundsPolicy policy{OutOfBoundsMode::FILL, fill_value};
  auto evaluator = factory.CreatePointEvaluator(cv, policy);
  CheckFillMode(*evaluator, field_data, fill_value, outside_pts);
}

// Check a mix of inside/outside points. inside points verified against func,
// outside points against fill_value.
template <typename Factory, typename ExecutionSpace = DefaultExecutionSpace,
          typename Func>
void CheckEvaluationWithFill(const Factory& factory,
                             FieldData<Real>& field_data,
                             const std::vector<Real>& pts,
                             const std::vector<bool>& is_inside, Func func,
                             Real fill_value, double abs_tol = 1e-10)
{
  int n = static_cast<int>(pts.size()) / 2;
  REQUIRE(static_cast<int>(is_inside.size()) == n);

  Rank2View<const Real, HostMemorySpace> coords_view(pts.data(), n, 2);
  CoordinateView<HostMemorySpace> cv{factory.GetCoordinateSystem(), coords_view};
  OutOfBoundsPolicy policy{OutOfBoundsMode::FILL, fill_value};
  auto evaluator = factory.CreatePointEvaluator(cv, policy);

  std::vector<Real> eval(n);
  Rank2View<Real, HostMemorySpace> out(eval.data(), n, 1);
  evaluator->Evaluate(field_data, out);

  auto expected = EvaluateReferenceFunction<ExecutionSpace>(pts, func);

  for (int i = 0; i < n; ++i) {
    INFO("Point " << i << " (" << pts[2*i] << ", " << pts[2*i+1] << ")"
         << " got=" << eval[i]);
    if (is_inside[i]) {
      INFO(" expected=" << expected[i]);
      REQUIRE(eval[i] == Catch::Approx(expected[i]).margin(abs_tol));
    } else {
      REQUIRE(eval[i] == fill_value);
    }
  }
}

} // namespace pcms::test

#endif // PCMS_TEST_FIELD_TEST_UTILS_H
