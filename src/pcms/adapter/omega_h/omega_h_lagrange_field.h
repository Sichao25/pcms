#ifndef PCMS_ADAPTER_OMEGA_H_LAGRANGE_FIELD_H
#define PCMS_ADAPTER_OMEGA_H_LAGRANGE_FIELD_H

#include "pcms/adapter/omega_h/omega_h_lagrange_layout.h"
#include "pcms/field.h"
#include "pcms/localization/point_search.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/profile.h"

#include <memory>
#include <variant>

namespace pcms {
struct OmegaHLagrangeLocHint {
  // Per valid query point:
  Kokkos::View<LO *, HostMemorySpace> elem_ids; // containing element
  Kokkos::View<Real **, HostMemorySpace> bary; // [n_valid x (dim+1)]
  Kokkos::View<LO *, HostMemorySpace> orig_indices; // original query index
  // Points that fell outside the mesh:
  Kokkos::View<LO *, HostMemorySpace> missing_indices;
  OutOfBoundsMode mode;
};

namespace detail {
template<int Dim> OmegaHLagrangeLocHint BuildLagrangeLocHint(
  int mesh_dim,
  Kokkos::View<typename PointLocalizationSearch<Dim>::Result *,
               HostMemorySpace> results_h,
  OutOfBoundsMode mode) {
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
    PCMS_ALWAYS_ASSERT(missing.empty() &&
      "Points found outside mesh domain");
  } else if (mode == OutOfBoundsMode::NEAREST_BOUNDARY) {
    PCMS_ALWAYS_ASSERT(false &&
      "NEAREST_BOUNDARY mode not yet implemented");
  }

  LO nv = static_cast<LO>(valid.size());
  LO nm = static_cast<LO>(missing.size());

  Kokkos::View<LO *, HostMemorySpace> elem_ids("elem_ids", nv);
  Kokkos::View<Real **, HostMemorySpace> bary("bary", nv, mesh_dim + 1);
  Kokkos::View<LO *, HostMemorySpace> orig_indices("orig_indices", nv);
  Kokkos::View<LO *, HostMemorySpace> missing_indices("missing_indices", nm);

  for (LO k = 0; k < nv; ++k) {
    LO i = valid[k];
    elem_ids(k) = results_h(i).element_id;
    orig_indices(k) = i;
    for (int d = 0; d <= mesh_dim; ++d)
      bary(k, d) = results_h(i).parametric_coords[d];
  }
  for (LO k = 0; k < nm; ++k)
    missing_indices(k) = missing[k];

  return OmegaHLagrangeLocHint{
    elem_ids, bary, orig_indices, missing_indices,
    mode
  };
}
} // namespace detail

namespace detail {
inline std::variant<GridPointSearch2D, GridPointSearch3D> MakeSearch(
  Omega_h::Mesh &mesh) {
  if (mesh.dim() == 2)
    return GridPointSearch2D(mesh, 10, 10);
  else if (mesh.dim() == 3)
    return GridPointSearch3D(mesh, 10, 10, 10);
  throw std::invalid_argument(
    "OmegaHLagrangeField: only 2D and 3D meshes are supported");
}
} // namespace detail

template<typename T> class OmegaHLagrangeField : public FieldT<T> {
public:
  explicit OmegaHLagrangeField(
    std::shared_ptr<const OmegaHLagrangeLayout> layout)
    : layout_(std::move(layout)),
      mesh_(layout_->GetMesh()),
      search_(detail::MakeSearch(mesh_)),
      dof_holder_data_("dof_holder_data",
                       static_cast<size_t>(layout_->OwnedSize())) {
  }

  LocalizationHint GetLocalizationHint(
    CoordinateView<HostMemorySpace> coordinate_view) const override {
    PCMS_FUNCTION_TIMER;
    if (coordinate_view.GetCoordinateSystem() !=
      layout_->GetDOFHolderCoordinates().GetCoordinateSystem()) {
      throw pcms_error("Coordinate system mismatch");
    }

    auto coords = coordinate_view.GetCoordinates();
    LO n_pts = static_cast<LO>(coords.extent(0));
    int mesh_dim = mesh_.dim();

    auto hint =
        std::make_shared<OmegaHLagrangeLocHint>(std::visit(
          [&](auto &search) {
            using SearchT = std::decay_t<decltype(search)>;
            constexpr int Dim = SearchT::DIM;

            Kokkos::View<Real *[Dim]> coords_d("coords_d", n_pts);
            auto coords_h = Kokkos::View<const Real **, HostMemorySpace>(
              coords.data_handle(),
              n_pts,
              Dim);
            deep_copy_mismatch_layouts(coords_d, coords_h);

            auto results_d = search(coords_d);
            Kokkos::View<typename PointLocalizationSearch<Dim>::Result *,
                         HostMemorySpace>
                results_h("results_h", results_d.size());
            Kokkos::deep_copy(results_h, results_d);

            return detail::BuildLagrangeLocHint<Dim>(mesh_dim,
                                                     results_h,
                                                     this->out_of_bounds_mode_);
          },
          search_));

    return LocalizationHint{hint};
  }

  void Evaluate(LocalizationHint location,
                FieldDataView<T, HostMemorySpace> results) const override {
    PCMS_FUNCTION_TIMER;
    if (results.GetCoordinateSystem() !=
      layout_->GetDOFHolderCoordinates().GetCoordinateSystem()) {
      throw pcms_error("Coordinate system mismatch");
    }

    const auto &hint =
        *reinterpret_cast<OmegaHLagrangeLocHint *>(location.data.get());
    auto values = results.GetValues();
    LO n_valid = static_cast<LO>(hint.elem_ids.size());

    if (layout_->GetOrder() == 0) {
      // One DOF per element — return the element's value directly
      for (LO k = 0; k < n_valid; ++k)
        values[hint.orig_indices(k)] = dof_holder_data_(hint.elem_ids(k));
    } else {
      // Order-1: barycentric interpolation over element vertices
      int mesh_dim = mesh_.dim();
      int nvpe = mesh_dim + 1; // vertices per element (simplex)
      auto elem_verts = Omega_h::HostRead<Omega_h::LO>(mesh_.ask_elem_verts());
      for (LO k = 0; k < n_valid; ++k) {
        LO elem = hint.elem_ids(k);
        T val = T{};
        for (int v = 0; v < nvpe; ++v)
          val += static_cast<T>(hint.bary(k, v)) *
              dof_holder_data_(elem_verts[elem * nvpe + v]);
        values[hint.orig_indices(k)] = val;
      }
    }

    if (hint.mode == OutOfBoundsMode::FILL) {
      T fill_val = static_cast<T>(this->fill_value_);
      for (LO k = 0; k < static_cast<LO>(hint.missing_indices.size()); ++k)
        values[hint.missing_indices(k)] = fill_val;
    }
  }

  void EvaluateGradient(FieldDataView<T, HostMemorySpace> /*unused*/) override {
    throw pcms_error(
      "EvaluateGradient not implemented for OmegaHLagrangeField");
  }

  bool CanEvaluateGradient() override { return false; }

  Rank1View<const T, HostMemorySpace> GetDOFHolderData() const override {
    return make_const_array_view(dof_holder_data_);
  }

  void SetDOFHolderData(Rank1View<const T, HostMemorySpace> data) override {
    PCMS_ALWAYS_ASSERT(static_cast<LO>(data.size()) ==
      layout_->GetNumOwnedDofHolder() *
      layout_->GetNumComponents());
    for (size_t i = 0; i < data.size(); ++i)
      dof_holder_data_(i) = data[i];
  }

  const FieldLayout &GetLayout() const override { return *layout_; }

  int Serialize(
    Rank1View<T, pcms::HostMemorySpace> buffer,
    Rank1View<const pcms::LO, pcms::HostMemorySpace> permutation) const override {
    PCMS_FUNCTION_TIMER;
    auto owned = layout_->GetOwned();
    LO n = static_cast<LO>(dof_holder_data_.size());
    if (buffer.size() > 0) {
      for (LO i = 0; i < n; ++i) {
        if (owned[i])
          buffer[permutation[i]] = dof_holder_data_(i);
      }
    }
    return n;
  }

  void Deserialize(
    Rank1View<const T, pcms::HostMemorySpace> buffer,
    Rank1View<const pcms::LO, pcms::HostMemorySpace> permutation) override {
    PCMS_FUNCTION_TIMER;
    auto owned = layout_->GetOwned();
    LO n = static_cast<LO>(dof_holder_data_.size());
    for (LO i = 0; i < n; ++i) {
      if (owned[i])
        dof_holder_data_(i) = buffer[permutation[i]];
    }
  }

  ~OmegaHLagrangeField() noexcept = default;

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  Omega_h::Mesh &mesh_;
  std::variant<GridPointSearch2D, GridPointSearch3D> search_;
  Kokkos::View<T *, HostMemorySpace> dof_holder_data_;
};
} // namespace pcms
#endif // PCMS_ADAPTER_OMEGA_H_LAGRANGE_FIELD_H
