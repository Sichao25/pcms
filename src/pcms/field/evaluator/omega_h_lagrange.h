#ifndef PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H
#define PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H

#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/localization/point_search.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/types.h"

#include <Omega_h_mesh.hpp>
#include <Kokkos_Core.hpp>
#include <memory>
#include <stdexcept>
#include <variant>

namespace pcms
{

// ---------------------------------------------------------------------------
// Localization hint — computed once per query point set, reused across Evaluate
// ---------------------------------------------------------------------------
struct OmegaHLagrangeLocHint
{
  Kokkos::View<LO*, HostMemorySpace> elem_ids;      // containing element (valid pts)
  Kokkos::View<Real**, HostMemorySpace> bary;        // [n_valid x (dim+1)]
  Kokkos::View<LO*, HostMemorySpace> orig_indices;   // original query index
  Kokkos::View<LO*, HostMemorySpace> missing_indices;
  OutOfBoundsMode mode;
};

namespace detail
{

template <int Dim>
OmegaHLagrangeLocHint BuildLagrangeLocHint(
  int mesh_dim,
  Kokkos::View<typename PointLocalizationSearch<Dim>::Result*,
               HostMemorySpace> results_h,
  OutOfBoundsMode mode)
{
  LO n = static_cast<LO>(results_h.size());
  std::vector<LO> valid, missing;
  for (LO i = 0; i < n; ++i) {
    bool out = (static_cast<int>(results_h(i).dimensionality) != mesh_dim) ||
               (results_h(i).element_id < 0);
    if (out)
      missing.push_back(i);
    else
      valid.push_back(i);
  }

  if (mode == OutOfBoundsMode::ERROR) {
    PCMS_ALWAYS_ASSERT(missing.empty() && "Points found outside mesh domain");
  } else if (mode == OutOfBoundsMode::NEAREST_BOUNDARY) {
    PCMS_ALWAYS_ASSERT(false && "NEAREST_BOUNDARY mode not yet implemented");
  }

  LO nv = static_cast<LO>(valid.size());
  LO nm = static_cast<LO>(missing.size());

  Kokkos::View<LO*, HostMemorySpace> elem_ids("elem_ids", nv);
  Kokkos::View<Real**, HostMemorySpace> bary("bary", nv, mesh_dim + 1);
  Kokkos::View<LO*, HostMemorySpace> orig_indices("orig_indices", nv);
  Kokkos::View<LO*, HostMemorySpace> missing_indices("missing_indices", nm);

  for (LO k = 0; k < nv; ++k) {
    LO i = valid[k];
    elem_ids(k) = results_h(i).element_id;
    orig_indices(k) = i;
    for (int d = 0; d <= mesh_dim; ++d)
      bary(k, d) = results_h(i).parametric_coords[d];
  }
  for (LO k = 0; k < nm; ++k)
    missing_indices(k) = missing[k];

  return OmegaHLagrangeLocHint{elem_ids, bary, orig_indices, missing_indices,
                                mode};
}

inline std::variant<GridPointSearch2D, GridPointSearch3D> MakeSearch(
  Omega_h::Mesh& mesh)
{
  if (mesh.dim() == 2)
    return GridPointSearch2D(mesh, 10, 10);
  else if (mesh.dim() == 3)
    return GridPointSearch3D(mesh, 10, 10, 10);
  throw std::invalid_argument(
    "OmegaHLagrangeEvaluatorFactory: only 2D and 3D meshes are supported");
}

} // namespace detail

// ---------------------------------------------------------------------------
// PointEvaluator
// ---------------------------------------------------------------------------

// OmegaHLagrangePointEvaluator<T> implements PointEvaluator<T> for simplex
// meshes backed by Omega_h. Localization results (element IDs, barycentric
// coordinates) are computed once at construction and cached for repeated
// Evaluate calls.
//
// Output shape: [num_query_points][num_components].
template <typename T>
class OmegaHLagrangePointEvaluator : public PointEvaluator<T>
{
public:
  OmegaHLagrangePointEvaluator(
    std::shared_ptr<const OmegaHLagrangeLayout> layout,
    OmegaHLagrangeLocHint hint, Real fill_value)
    : layout_(std::move(layout)),
      hint_(std::move(hint)),
      fill_value_(fill_value)
  {
  }

  void Evaluate(const Field<T>& field,
                Rank2View<T, HostMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    auto dof_data = field.GetDOFHolderDataHost();
    LO n_valid = static_cast<LO>(hint_.elem_ids.size());
    int n_comp = layout_->GetNumComponents();

    PCMS_ALWAYS_ASSERT(values.extent(1) == static_cast<size_t>(n_comp));

    if (layout_->GetOrder() == 0) {
      for (LO k = 0; k < n_valid; ++k) {
        LO orig = hint_.orig_indices(k);
        LO elem = hint_.elem_ids(k);
        for (int c = 0; c < n_comp; ++c) {
          values(orig, c) = dof_data[elem * n_comp + c];
        }
      }
    } else {
      // Order-1: barycentric interpolation over element vertices
      Omega_h::Mesh& mesh = const_cast<Omega_h::Mesh&>(layout_->GetMesh());
      int mesh_dim = mesh.dim();
      int nvpe = mesh_dim + 1;
      auto elem_verts = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());

      for (LO k = 0; k < n_valid; ++k) {
        LO orig = hint_.orig_indices(k);
        LO elem = hint_.elem_ids(k);
        for (int c = 0; c < n_comp; ++c) {
          T val = T{};
          for (int v = 0; v < nvpe; ++v) {
            LO vert = elem_verts[elem * nvpe + v];
            val +=
              static_cast<T>(hint_.bary(k, v)) * dof_data[vert * n_comp + c];
          }
          values(orig, c) = val;
        }
      }
    }

    if (hint_.mode == OutOfBoundsMode::FILL) {
      T fill_val = static_cast<T>(fill_value_);
      for (LO k = 0; k < static_cast<LO>(hint_.missing_indices.size()); ++k) {
        LO orig = hint_.missing_indices(k);
        for (int c = 0; c < n_comp; ++c) {
          values(orig, c) = fill_val;
        }
      }
    }
  }

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  void Evaluate(const Field<T>& field,
                Rank2View<T, DeviceMemorySpace> values) const override
  {
    // TODO: rewrite this after switch backend to device-native evaluation
    auto values_host_view = Kokkos::View<T**, HostMemorySpace>(values.data_handle(), values.extent(0), values.extent(1));
    auto values_host = Rank2View<T, HostMemorySpace>(values_host_view.data(), values_host_view.extent(0), values_host_view.extent(1));
    Evaluate(field, values_host);
    auto values_device_view = Kokkos::View<T**, DeviceMemorySpace>(values.data_handle(), values.extent(0), values.extent(1));
    Kokkos::deep_copy(values_device_view, values_host_view);
    Kokkos::parallel_for("CopyHostToDevice", Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, values.extent(0)),
                         KOKKOS_LAMBDA(int i) {
                           for (int j = 0; j < values.extent(1); ++j) {
                             values(i, j) = values_host(i, j);
                           }
                         });



  }
#endif

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  OmegaHLagrangeLocHint hint_;
  Real fill_value_;
};

// ---------------------------------------------------------------------------
// EvaluatorFactory
// ---------------------------------------------------------------------------

// OmegaHLagrangeEvaluatorFactory<T> implements FieldEvaluatorFactory<T> for
// simplex meshes backed by Omega_h. It owns the spatial search structure and
// creates OmegaHLagrangePointEvaluator instances on demand.
template <typename T>
class OmegaHLagrangeEvaluatorFactory : public FieldEvaluatorFactory<T>
{
public:
  explicit OmegaHLagrangeEvaluatorFactory(
    std::shared_ptr<const OmegaHLagrangeLayout> layout)
    : layout_(std::move(layout)),
      search_(detail::MakeSearch(layout_->GetMesh()))
  {
  }

  const FieldLayout& GetLayout() const override { return *layout_; }

  CoordinateSystem GetCoordinateSystem() const override
  {
    return layout_->GetDOFHolderCoordinatesHost().GetCoordinateSystem();
  }

  bool HasDOFHolderCoordinates() const override { return true; }

  CoordinateView<HostMemorySpace> GetDOFHolderCoordinatesHost() const override
  {
    return layout_->GetDOFHolderCoordinatesHost();
  }

  bool SupportsNearestBoundary() const override { return false; }

  std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    const EvaluationRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto coords = request.coords;
    const auto policy = request.policy;
    if (coords.GetCoordinateSystem() != GetCoordinateSystem()) {
      throw pcms_error(
        "OmegaHLagrangeEvaluatorFactory: coordinate system mismatch");
    }
    if (policy.mode == OutOfBoundsMode::NEAREST_BOUNDARY) {
      throw pcms_error(
        "OmegaHLagrangeEvaluatorFactory: NearestBoundary is not supported");
    }

    auto raw_coords = coords.GetCoordinates();
    LO n_pts = static_cast<LO>(raw_coords.extent(0));
    int mesh_dim = layout_->GetMesh().dim();

    OmegaHLagrangeLocHint hint = std::visit(
      [&](auto& search) {
        using SearchT = std::decay_t<decltype(search)>;
        constexpr int Dim = SearchT::DIM;

        Kokkos::View<Real**> coords_d("coords_d", n_pts, Dim);
        auto coords_h = Kokkos::View<const Real**, HostMemorySpace>(
          raw_coords.data_handle(), n_pts, Dim);
        DeepCopyMismatchLayouts(coords_d, coords_h);

        auto results_d = search(coords_d);
        Kokkos::View<typename PointLocalizationSearch<Dim>::Result*,
                     HostMemorySpace>
          results_h("results_h", results_d.size());
        Kokkos::deep_copy(results_h, results_d);

        return detail::BuildLagrangeLocHint<Dim>(mesh_dim, results_h,
                                                 policy.mode);
      },
      search_);

    return std::make_unique<OmegaHLagrangePointEvaluator<T>>(
      layout_, std::move(hint), policy.fill_value);
  }

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinatesDevice() const override
  {
    throw pcms_error(
      "OmegaHLagrangeEvaluatorFactory: GetDOFHolderCoordinatesDevice not yet implemented");
  }

  std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    CoordinateView<DeviceMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const override
  {
    throw pcms_error(
      "OmegaHLagrangeEvaluatorFactory: CreatePointEvaluator with device coordinates not yet implemented");
  }
#endif

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  mutable std::variant<GridPointSearch2D, GridPointSearch3D> search_;
};

} // namespace pcms

#endif // PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H
