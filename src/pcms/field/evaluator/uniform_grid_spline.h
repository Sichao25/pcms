#ifndef PCMS_UNIFORM_GRID_SPLINE_EVALUATOR_FACTORY_H
#define PCMS_UNIFORM_GRID_SPLINE_EVALUATOR_FACTORY_H

#include "pcms/field/evaluator/spline_interpolator.hpp"
#include "pcms/field/evaluator/uniform_grid.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/layout/uniform_grid.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/types.h"

#include <memory>

namespace pcms
{

class UniformGridSplinePointEvaluator2D : public PointEvaluator<Real>
{
public:
  UniformGridSplinePointEvaluator2D(
    std::shared_ptr<const UniformGridFieldLayout<2>> layout,
    UniformGridFieldLocalizationHint<2> hint, Real fill_value)
    : layout_(std::move(layout)),
      grid_(layout_->GetGrid()),
      hint_(std::move(hint)),
      fill_value_(fill_value),
      x_coords_("uniform_grid_spline_x", grid_.divisions[0] + 1),
      y_coords_("uniform_grid_spline_y", grid_.divisions[1] + 1)
  {
    Real dx = grid_.edge_length[0] / grid_.divisions[0];
    Real dy = grid_.edge_length[1] / grid_.divisions[1];
    for (LO ix = 0; ix <= grid_.divisions[0]; ++ix) {
      x_coords_(ix) = grid_.bot_left[0] + dx * ix;
    }
    for (LO iy = 0; iy <= grid_.divisions[1]; ++iy) {
      y_coords_(iy) = grid_.bot_left[1] + dy * iy;
    }
  }

  void Evaluate(const FieldData<Real>& field,
                Rank2View<Real, HostMemorySpace> values) const override
  {
    if (&field.GetLayout() != layout_.get()) {
      throw pcms_error(
        "UniformGridSplinePointEvaluator2D::Evaluate: FieldData layout does "
        "not match the evaluator layout");
    }

    LO num_points = static_cast<LO>(hint_.coordinates_.extent(0));
    PCMS_ALWAYS_ASSERT(values.extent(0) == static_cast<size_t>(num_points));
    PCMS_ALWAYS_ASSERT(values.extent(1) == 1);

    auto dof_data = field.GetDOFHolderDataHost();
    LO nx = grid_.divisions[0] + 1;
    LO ny = grid_.divisions[1] + 1;
    PCMS_ALWAYS_ASSERT(static_cast<LO>(dof_data.size()) == nx * ny);

    Kokkos::View<Real*, HostMemorySpace> spline_values(
      "uniform_grid_spline_values", static_cast<size_t>(nx * ny));
    for (LO iy = 0; iy < ny; ++iy) {
      for (LO ix = 0; ix < nx; ++ix) {
        spline_values(ix * ny + iy) = dof_data[iy * nx + ix];
      }
    }

    if (hint_.num_out_of_bounds_ == static_cast<size_t>(num_points) &&
        hint_.mode_ == OutOfBoundsMode::FILL) {
      for (LO i = 0; i < num_points; ++i) {
        values(i, 0) = fill_value_;
      }
      return;
    }

    LO num_in_bounds = static_cast<LO>(num_points - hint_.num_out_of_bounds_);
    Kokkos::View<Real*, HostMemorySpace> eval_x("uniform_grid_spline_eval_x",
                                                num_in_bounds);
    Kokkos::View<Real*, HostMemorySpace> eval_y("uniform_grid_spline_eval_y",
                                                num_in_bounds);
    Kokkos::View<LO*, HostMemorySpace> in_bounds_indices(
      "uniform_grid_spline_in_bounds_indices", num_in_bounds);

    LO in_bounds_idx = 0;
    for (LO i = 0; i < num_points; ++i) {
      if (hint_.is_out_of_bounds_[i]) {
        continue;
      }
      eval_x(in_bounds_idx) = hint_.coordinates_(i, 0);
      eval_y(in_bounds_idx) = hint_.coordinates_(i, 1);
      in_bounds_indices(in_bounds_idx) = i;
      ++in_bounds_idx;
    }
    PCMS_ALWAYS_ASSERT(in_bounds_idx == num_in_bounds);

    Kokkos::View<Real**, HostMemorySpace> spline_output(
      "uniform_grid_spline_output", num_in_bounds, 1);
    Kokkos::View<Real*, HostMemorySpace> empty_bc(
      "uniform_grid_spline_empty_bc", 0);

    CompactBiCubicSplineInterpolator<Real, HostMemorySpace> interpolator(
      Rank1View<Real, HostMemorySpace>(x_coords_.data(), x_coords_.extent(0)),
      Rank1View<Real, HostMemorySpace>(y_coords_.data(), y_coords_.extent(0)),
      Rank1View<Real, HostMemorySpace>(spline_values.data(),
                                       spline_values.extent(0)),
      BoundaryCondition::NOT_A_KNOT,
      Rank1View<Real, HostMemorySpace>(empty_bc.data(), empty_bc.extent(0)),
      BoundaryCondition::NOT_A_KNOT,
      Rank1View<Real, HostMemorySpace>(empty_bc.data(), empty_bc.extent(0)),
      BoundaryCondition::NOT_A_KNOT,
      Rank1View<Real, HostMemorySpace>(empty_bc.data(), empty_bc.extent(0)),
      BoundaryCondition::NOT_A_KNOT,
      Rank1View<Real, HostMemorySpace>(empty_bc.data(), empty_bc.extent(0)));
    interpolator.evaluate(
      Rank1View<Real, HostMemorySpace>(eval_x.data(), eval_x.extent(0)),
      Rank1View<Real, HostMemorySpace>(eval_y.data(), eval_y.extent(0)),
      Rank2View<Real, HostMemorySpace>(spline_output.data(),
                                       spline_output.extent(0), 1));

    if (hint_.mode_ == OutOfBoundsMode::FILL) {
      for (LO i = 0; i < num_points; ++i) {
        values(i, 0) = hint_.is_out_of_bounds_[i] ? fill_value_ : 0.0;
      }
    }

    for (LO i = 0; i < num_in_bounds; ++i) {
      values(in_bounds_indices(i), 0) = spline_output(i, 0);
    }
  }

private:
  std::shared_ptr<const UniformGridFieldLayout<2>> layout_;
  const UniformGrid<2>& grid_;
  UniformGridFieldLocalizationHint<2> hint_;
  Real fill_value_;
  Kokkos::View<Real*, HostMemorySpace> x_coords_;
  Kokkos::View<Real*, HostMemorySpace> y_coords_;
};

class UniformGridSplineEvaluatorFactory2D : public FieldEvaluatorFactory<Real>
{
public:
  explicit UniformGridSplineEvaluatorFactory2D(
    std::shared_ptr<const UniformGridFieldLayout<2>> layout)
    : layout_(std::move(layout)), grid_(layout_->GetGrid())
  {
  }

  const FieldLayout& GetLayout() const override { return *layout_; }

  CoordinateSystem GetCoordinateSystem() const override
  {
    return layout_->GetDOFHolderCoordinates().GetCoordinateSystem();
  }

  bool HasDOFHolderCoordinates() const override { return true; }

  CoordinateView<HostMemorySpace> GetDOFHolderCoordinatesHost() const override
  {
    return layout_->GetDOFHolderCoordinates();
  }

  bool SupportsNearestBoundary() const override { return false; }

  std::unique_ptr<PointEvaluator<Real>> CreatePointEvaluator(
    CoordinateView<HostMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const override
  {
    if (coords.GetCoordinateSystem() != GetCoordinateSystem()) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory2D: coordinate system mismatch");
    }
    if (policy.mode == OutOfBoundsMode::NEAREST_BOUNDARY) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory2D: nearest-boundary evaluation is "
        "not supported");
    }
    if (layout_->GetOrder() != 1) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory2D: spline evaluation requires an "
        "order-1 uniform-grid layout");
    }
    if (layout_->GetNumComponents() != 1) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory2D: spline evaluation only supports "
        "single-component fields");
    }

    auto coordinates = coords.GetCoordinates();
    LO num_points = static_cast<LO>(coordinates.extent(0));

    Kokkos::View<LO*, HostMemorySpace> cell_indices("cell_indices", num_points);
    Kokkos::View<Real**, HostMemorySpace> coords_copy("coords_copy", num_points,
                                                      2);
    Kokkos::View<bool*, HostMemorySpace> is_out_of_bounds("is_out_of_bounds",
                                                          num_points);
    size_t num_out_of_bounds = 0;

    for (LO i = 0; i < num_points; ++i) {
      Omega_h::Vector<2> point;
      for (unsigned d = 0; d < 2; ++d) {
        point[d] = coordinates(i, d);
        coords_copy(i, d) = coordinates(i, d);
      }

      bool out_of_bounds = !grid_.IsPointInBounds(point);
      is_out_of_bounds(i) = out_of_bounds;
      if (out_of_bounds) {
        ++num_out_of_bounds;
        if (policy.mode == OutOfBoundsMode::ERROR) {
          throw pcms_error(
            "UniformGridSplineEvaluatorFactory2D: point found outside "
            "uniform grid domain");
        }
      }

      cell_indices(i) = grid_.ClosestCellID(point);
    }

    UniformGridFieldLocalizationHint<2> hint(cell_indices, coords_copy,
                                             policy.mode, is_out_of_bounds,
                                             num_out_of_bounds);
    return std::make_unique<UniformGridSplinePointEvaluator2D>(
      layout_, std::move(hint), policy.fill_value);
  }

private:
  std::shared_ptr<const UniformGridFieldLayout<2>> layout_;
  const UniformGrid<2>& grid_;
};

} // namespace pcms

#endif // PCMS_UNIFORM_GRID_SPLINE_EVALUATOR_FACTORY_H
