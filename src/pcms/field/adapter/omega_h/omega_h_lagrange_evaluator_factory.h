#ifndef PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H
#define PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H

#include "omega_h_lagrange_common.h"
#include "omega_h_lagrange_layout.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/profile.h"

#include <memory>
#include <variant>

namespace pcms
{

// OmegaHLagrangePointEvaluator<T> implements PointEvaluator<T> for simplex
// meshes backed by Omega_h. Localization results (element IDs, barycentric
// coordinates) are computed once at construction and cached for repeated
// Evaluate calls.
//
// Evaluation logic is taken directly from OmegaHLagrangeField::Evaluate.
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

  void Evaluate(const FieldData<T>& field,
                Rank2View<T, HostMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    auto dof_data = field.GetDOFHolderDataHost();
    LO n_valid = static_cast<LO>(hint_.elem_ids.size());
    LO n_pts = static_cast<LO>(values.extent(0));
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

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  OmegaHLagrangeLocHint hint_;
  Real fill_value_;
};

// OmegaHLagrangeEvaluatorFactory<T> implements FieldEvaluatorFactory<T> for
// simplex meshes backed by Omega_h. It owns the spatial search structure and
// creates OmegaHLagrangePointEvaluator instances on demand.
//
// Localization logic is taken directly from
// OmegaHLagrangeField::GetLocalizationHint.
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
    return layout_->GetDOFHolderCoordinates().GetCoordinateSystem();
  }

  bool HasDOFHolderCoordinates() const override { return true; }

  CoordinateView<HostMemorySpace> GetDOFHolderCoordinatesHost() const override
  {
    return layout_->GetDOFHolderCoordinates();
  }

  bool SupportsNearestBoundary() const override { return false; }

  std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    CoordinateView<HostMemorySpace> coords,
    OutOfBoundsPolicy policy) const override
  {
    PCMS_FUNCTION_TIMER;
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

        Kokkos::View<Real* [Dim]> coords_d("coords_d", n_pts);
        auto coords_h = Kokkos::View<const Real**, HostMemorySpace>(
          raw_coords.data_handle(), n_pts, Dim);
        deep_copy_mismatch_layouts(coords_d, coords_h);

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

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  mutable std::variant<GridPointSearch2D, GridPointSearch3D> search_;
};

} // namespace pcms

#endif // PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H
