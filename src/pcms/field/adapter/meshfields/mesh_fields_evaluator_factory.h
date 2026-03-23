#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H

#include "mesh_fields_backend.h"
#include "mesh_fields_field_data.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/arrays.h"
#include <memory>

namespace pcms
{

// MeshFieldsPointEvaluator<T> implements PointEvaluator<T> for MeshFields-backed
// simplex meshes. Each Evaluate call loads DOF data from the FieldData argument
// into the MeshFieldBackend's shape_field_ via SetData, then calls evaluate().
template <typename T>
class MeshFieldsPointEvaluator : public PointEvaluator<T>
{
public:
  MeshFieldsPointEvaluator(
    std::shared_ptr<const MeshFieldsAdapterLayout> layout,
    MeshFieldsAdapter2LocalizationHint hint, Real fill_value)
    : layout_(std::move(layout)),
      hint_(std::move(hint)),
      fill_value_(fill_value)
  {
  }

  void Evaluate(const FieldData<T>& field,
                Rank2View<T, HostMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    if (&field.GetLayout() != layout_.get()) {
      throw pcms_error(
        "MeshFieldsPointEvaluator::Evaluate: FieldData layout does not match "
        "the evaluator layout");
    }
    PCMS_ALWAYS_ASSERT(values.extent(0) ==
                       hint_.coordinates_.extent(0) + hint_.num_missing_);
    PCMS_ALWAYS_ASSERT(values.extent(1) ==
                       static_cast<size_t>(layout_->GetNumComponents()));
    auto const* mesh_field_data =
      dynamic_cast<const MeshFieldsFieldData<T>*>(&field);
    if (!mesh_field_data) {
      throw pcms_error(
        "MeshFieldsPointEvaluator::Evaluate: incompatible FieldData type");
    }

    Kokkos::View<Real**> coordinates_d(
      "coordinates_d", hint_.coordinates_.extent(0),
      hint_.coordinates_.extent(1));
    deep_copy_mismatch_layouts(coordinates_d, hint_.coordinates_);

    Kokkos::View<LO*> offsets_d("offsets_d", hint_.offsets_.extent(0));
    Kokkos::deep_copy(offsets_d, hint_.offsets_);

    auto eval_results = mesh_field_data->GetMeshFieldBackend()->evaluate(
      coordinates_d, offsets_d);
    Kokkos::View<T**, HostMemorySpace> eval_results_h(
      "eval_results_h", eval_results.extent(0), eval_results.extent(1));
    deep_copy_mismatch_layouts(eval_results_h, eval_results);

    Kokkos::parallel_for(
      "CopyEvalResultsToValues",
      Kokkos::RangePolicy<HostMemorySpace::execution_space>(
        0, eval_results_h.extent(0)),
      KOKKOS_LAMBDA(LO i) {
        values(hint_.indices_(i), 0) = eval_results_h(i, 0);
      });

    if (hint_.num_missing_ > 0 && hint_.mode_ == OutOfBoundsMode::FILL) {
      T fill_val = static_cast<T>(fill_value_);
      Kokkos::parallel_for(
        "FillMissingValues",
        Kokkos::RangePolicy<HostMemorySpace::execution_space>(
          0, hint_.num_missing_),
        KOKKOS_LAMBDA(LO i) {
          values(hint_.missing_indices_(i), 0) = fill_val;
        });
    }
  }

private:
  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  MeshFieldsAdapter2LocalizationHint hint_;
  Real fill_value_;
};

// MeshFieldsEvaluatorFactory<T> implements FieldEvaluatorFactory<T> for
// MeshFields-backed simplex meshes. It owns the spatial search structure and
// the MeshFieldBackend (which holds the internal shape field).
template <typename T>
class MeshFieldsEvaluatorFactory : public FieldEvaluatorFactory<T>
{
public:
  explicit MeshFieldsEvaluatorFactory(
    std::shared_ptr<const MeshFieldsAdapterLayout> layout)
    : layout_(std::move(layout)),
      mesh_(layout_->GetMesh()),
      search_(mesh_, 10, 10)
  {
    if (mesh_.dim() == 3) {
      throw pcms_error(
        "MeshFieldsEvaluatorFactory does not support 3D meshes");
    }
    if (layout_->GetNumComponents() != 1) {
      throw pcms_error(
        "MeshFieldsEvaluatorFactory only supports single-component fields");
    }
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
    OutOfBoundsPolicy policy = {}) const override
  {
    PCMS_FUNCTION_TIMER;
    if (coords.GetCoordinateSystem() != GetCoordinateSystem()) {
      throw pcms_error(
        "MeshFieldsEvaluatorFactory: coordinate system mismatch");
    }
    if (policy.mode == OutOfBoundsMode::NEAREST_BOUNDARY) {
      throw pcms_error(
        "MeshFieldsEvaluatorFactory: NearestBoundary is not supported");
    }

    auto coordinates = coords.GetCoordinates();
    Kokkos::View<Real* [2]> coords_d("coords_d", coordinates.extent(0));
    auto coords_h = Kokkos::View<const Real**, HostMemorySpace>(
      coordinates.data_handle(), coordinates.extent(0),
      coordinates.extent(1));
    deep_copy_mismatch_layouts(coords_d, coords_h);

    auto results = search_(coords_d);
    Kokkos::View<GridPointSearch2D::Result*, HostMemorySpace> results_h(
      "results_h", results.size());
    Kokkos::deep_copy(results_h, results);

    MeshFieldsAdapter2LocalizationHint hint(mesh_, results_h, policy.mode);

    return std::make_unique<MeshFieldsPointEvaluator<T>>(
      layout_, std::move(hint), policy.fill_value);
  }

private:
  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  Omega_h::Mesh& mesh_;
  mutable GridPointSearch2D search_;
};

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H
