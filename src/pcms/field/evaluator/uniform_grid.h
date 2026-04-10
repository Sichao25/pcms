#ifndef PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H
#define PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H

#include "pcms/field/layout/uniform_grid.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/profile.h"
#include "pcms/transfer/linear_interpolant.hpp"
#include "pcms/transfer/multidimarray.hpp"

#include <memory>

namespace pcms
{

// ---------------------------------------------------------------------------
// Localization hint — computed once per query point set, reused across Evaluate
// ---------------------------------------------------------------------------
template <unsigned Dim>
struct UniformGridFieldLocalizationHint
{
  UniformGridFieldLocalizationHint(
    Kokkos::View<LO*, HostMemorySpace> cell_indices,
    Kokkos::View<Real**, HostMemorySpace> coordinates,
    OutOfBoundsMode mode,
    Kokkos::View<bool*, HostMemorySpace> is_out_of_bounds,
    size_t num_out_of_bounds)
    : cell_indices_(cell_indices),
      coordinates_(coordinates),
      mode_(mode),
      is_out_of_bounds_(is_out_of_bounds),
      num_out_of_bounds_(num_out_of_bounds)
  {
  }

  Kokkos::View<LO*, HostMemorySpace>   cell_indices_;
  Kokkos::View<Real**, HostMemorySpace> coordinates_;
  OutOfBoundsMode                       mode_;
  Kokkos::View<bool*, HostMemorySpace>  is_out_of_bounds_;
  size_t                                num_out_of_bounds_;
};

// UniformGridPointEvaluator<Dim> implements PointEvaluator<Real> for structured
// uniform grids. Localization results (cell indices, parametric coordinates)
// are computed once at construction and cached for repeated Evaluate calls.
//
// Evaluation logic is taken directly from UniformGridField::Evaluate.
// Output shape: [num_query_points][num_components].
// Preconditions:
//   - field is compatible with layout_.
//   - values has extents [num_query_points][num_components].
template <unsigned Dim = 2>
class UniformGridPointEvaluator : public PointEvaluator<Real>
{
public:
  UniformGridPointEvaluator(
    std::shared_ptr<const UniformGridFieldLayout<Dim>> layout,
    UniformGridFieldLocalizationHint<Dim> hint, Real fill_value)
    : layout_(std::move(layout)),
      grid_(layout_->GetGrid()),
      hint_(std::move(hint)),
      fill_value_(fill_value)
  {
  }

  void Evaluate(const Field<Real>& field,
                Rank2View<Real, HostMemorySpace> values) const override
  {
    auto values_device_view = Kokkos::View<Real**, DeviceMemorySpace>("values_device_view", values.extent(0), values.extent(1));
    auto values_device = Rank2View<Real, DeviceMemorySpace>(values_device_view.data(), values_device_view.extent(0), values_device_view.extent(1));
    Evaluate(field, values_device);
    auto values_host_view = Kokkos::View<Real**, HostMemorySpace>("values_host_view", values.extent(0), values.extent(1));
    DeepCopyMismatchLayouts(values_host_view, values_device_view);
    Kokkos::parallel_for("CopyDeviceToHost", Kokkos::RangePolicy<HostMemorySpace::execution_space>(0, values.extent(0)),
                         KOKKOS_LAMBDA(int i) {
                           for (int j = 0; j < values.extent(1); ++j) {
                             values(i, j) = values_host_view(i, j);
                           }
                         });
  }

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  void Evaluate(const Field<Real>& field,
                Rank2View<Real, DeviceMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    auto dof_data = field.GetDOFHolderData();
    LO num_points = static_cast<LO>(hint_.coordinates_.extent(0));
    int n_comp = layout_->GetNumComponents();

    PCMS_ALWAYS_ASSERT(values.extent(0) == static_cast<size_t>(num_points));
    PCMS_ALWAYS_ASSERT(values.extent(1) == static_cast<size_t>(n_comp));
    PCMS_ALWAYS_ASSERT(
      dof_data.size() == static_cast<size_t>(layout_->GetNumOwnedDofHolder() *
                                             layout_->GetNumComponents()));

    auto cell_indices = hint_.cell_indices_;
    auto cell_indices_device = Kokkos::View<LO*, DeviceMemorySpace>("cell_indices_device", num_points);
    Kokkos::deep_copy(cell_indices_device, cell_indices);

    auto coordinates = hint_.coordinates_;
    auto is_out_of_bounds = Kokkos::View<bool*, DeviceMemorySpace>("is_out_of_bounds", num_points);
    Kokkos::deep_copy(is_out_of_bounds, hint_.is_out_of_bounds_);


    if (layout_->GetOrder() == 0) {
      OutOfBoundsMode mode = hint_.mode_;
      Real fv = fill_value_;
      Kokkos::parallel_for(
        "evaluate_order0_device",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_points, n_comp}),
        KOKKOS_LAMBDA(const LO i, const int c) {
          if (is_out_of_bounds(i) && mode == OutOfBoundsMode::FILL) {
            values(i, c) = fv;
          } else {
            LO dof_idx = cell_indices_device(i);
            values(i, c) = dof_data[dof_idx * n_comp + c];
          }
        });
      return;
    }

    // Order-1: multilinear interpolation
    Kokkos::View<LO**, HostMemorySpace> cell_dim_indices("cell_dim_indices",
                                                         num_points, Dim);
    for (LO i = 0; i < num_points; ++i) {
      auto dim_idx = grid_.GetDimensionedIndex(cell_indices[i]);
      for (unsigned d = 0; d < Dim; ++d)
        cell_dim_indices(i, d) = dim_idx[d];
    }

    auto cell_divisions = grid_.divisions;
    IntVecView dimensions_view("dimensions", Dim);
    auto dimensions_host = Kokkos::create_mirror_view(dimensions_view);
    for (unsigned d = 0; d < Dim; ++d)
      dimensions_host(d) = cell_divisions[d] + 1;
    Kokkos::deep_copy(dimensions_view, dimensions_host);

    RealMatView parametric_coords("parametric_coords", num_points, Dim);
    auto param_host = Kokkos::create_mirror_view(parametric_coords);
    for (LO i = 0; i < num_points; ++i) {
      auto cell_bbox = grid_.GetCellBBOX(cell_indices[i]);
      for (unsigned d = 0; d < Dim; ++d) {
        Real coord = coordinates(i, d);
        Real cell_min = cell_bbox.center[d] - cell_bbox.half_width[d];
        Real cell_max = cell_bbox.center[d] + cell_bbox.half_width[d];
        param_host(i, d) = (coord - cell_min) / (cell_max - cell_min);
      }
    }
    Kokkos::deep_copy(parametric_coords, param_host);

    IntMatView cell_indices_interp("cell_indices_interp", num_points, Dim);
    auto cell_idx_host = Kokkos::create_mirror_view(cell_indices_interp);
    for (LO i = 0; i < num_points; ++i)
      for (unsigned d = 0; d < Dim; ++d)
        cell_idx_host(i, d) = cell_dim_indices(i, d);
    Kokkos::deep_copy(cell_indices_interp, cell_idx_host);

    // n_comp == 1 for now (multi-component path would need per-component calls)
    PCMS_ALWAYS_ASSERT(
      n_comp == 1 &&
      "UniformGridPointEvaluator: multi-component order-1 not yet supported");

    RealVecView values_interp("values_interp",
                              static_cast<size_t>(dof_data.size()));
    Kokkos::parallel_for(
      "copy_dof_data_to_values_interp",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, static_cast<LO>(dof_data.size())),
      KOKKOS_LAMBDA(const LO i) { values_interp(i) = dof_data[i]; });


    auto interpolator = RegularGridInterpolator(
      parametric_coords, values_interp, cell_indices_interp, dimensions_view);
    auto interpolated = interpolator.linear_interpolation();

    OutOfBoundsMode mode = hint_.mode_;
    Real fv = fill_value_;
    Kokkos::parallel_for(
      "evaluate_order1_device",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, num_points),
      KOKKOS_LAMBDA(const LO i) {
        if (is_out_of_bounds(i) && mode == OutOfBoundsMode::FILL) {
          values(i, 0) = fv;
        } else {
          values(i, 0) = interpolated(i);
        }
      });
  }
#endif

private:
  std::shared_ptr<const UniformGridFieldLayout<Dim>> layout_;
  const UniformGrid<Dim>& grid_;
  UniformGridFieldLocalizationHint<Dim> hint_;
  Real fill_value_;
};

// UniformGridEvaluatorFactory<Dim> implements FieldEvaluatorFactory<Real> for
// structured uniform grids. It owns the layout and creates
// UniformGridPointEvaluator instances on demand.
//
// Localization logic is taken directly from
// UniformGridField::GetLocalizationHint.
template <unsigned Dim = 2>
class UniformGridEvaluatorFactory : public FieldEvaluatorFactory<Real>
{
public:
  explicit UniformGridEvaluatorFactory(
    std::shared_ptr<const UniformGridFieldLayout<Dim>> layout)
    : layout_(std::move(layout)), grid_(layout_->GetGrid())
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

  bool SupportsNearestBoundary() const override { return true; }

  std::unique_ptr<PointEvaluator<Real>> CreatePointEvaluator(
    const EvaluationRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto coords = request.coords;
    const auto policy = request.policy;
    PCMS_FUNCTION_TIMER;
    if (coords.GetCoordinateSystem() != GetCoordinateSystem()) {
      throw pcms_error(
        "UniformGridEvaluatorFactory: coordinate system mismatch");
    }
    if (policy.mode == OutOfBoundsMode::NEAREST_BOUNDARY &&
        !SupportsNearestBoundary()) {
      throw pcms_error(
        "UniformGridEvaluatorFactory: nearest-boundary evaluation is not "
        "supported");
    }
    if (layout_->GetOrder() == 1 && layout_->GetNumComponents() != 1) {
      throw pcms_error(
        "UniformGridEvaluatorFactory: order-1 multi-component evaluation is "
        "not implemented");
    }

    auto coordinates = coords.GetCoordinates();
    LO num_points = static_cast<LO>(coordinates.extent(0));

    Kokkos::View<LO*, HostMemorySpace> cell_indices("cell_indices", num_points);
    Kokkos::View<Real**, DeviceMemorySpace> coordinates_device("coordinates_device", num_points, Dim);
    Kokkos::parallel_for("copy_coordinates_to_device", num_points, KOKKOS_LAMBDA(const LO i) {
      for (unsigned d = 0; d < Dim; ++d) {
        coordinates_device(i, d) = coordinates(i, d);
      }
    });
     Kokkos::View<Real**, HostMemorySpace> coords_copy("coords_copy", num_points, Dim);
    DeepCopyMismatchLayouts(coords_copy, coordinates_device);
    Kokkos::View<bool*, HostMemorySpace> is_out_of_bounds("is_out_of_bounds",
                                                          num_points);
    size_t num_out_of_bounds = 0;

    for (LO i = 0; i < num_points; ++i) {
      Omega_h::Vector<Dim> point;
      for (unsigned d = 0; d < Dim; ++d) {
        point[d] = coords_copy(i, d);
      }

      bool out_of_bounds = !grid_.IsPointInBounds(point);
      is_out_of_bounds[i] = out_of_bounds;
      if (out_of_bounds) {
        ++num_out_of_bounds;
        if (policy.mode == OutOfBoundsMode::ERROR) {
          throw pcms_error(
            "UniformGridEvaluatorFactory: point found outside uniform grid "
            "domain");
        }
      }

      cell_indices[i] = grid_.ClosestCellID(point);
    }

    UniformGridFieldLocalizationHint<Dim> hint(cell_indices, coords_copy,
                                               policy.mode, is_out_of_bounds,
                                               num_out_of_bounds);

    return std::make_unique<UniformGridPointEvaluator<Dim>>(
      layout_, std::move(hint), policy.fill_value);
  }

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinatesDevice() const override
  {
    return layout_->GetDOFHolderCoordinates();
  }
#endif

private:
  std::shared_ptr<const UniformGridFieldLayout<Dim>> layout_;
  const UniformGrid<Dim>& grid_;
};

using UniformGridEvaluatorFactory2D = UniformGridEvaluatorFactory<2>;

} // namespace pcms

#endif // PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H
