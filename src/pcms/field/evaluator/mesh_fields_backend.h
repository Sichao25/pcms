#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H

// Backend types shared between MeshFieldsAdapter2 and
// MeshFieldsEvaluatorFactory. Extracted here to avoid a circular include.

#include <Kokkos_Core.hpp>
#include <MeshField.hpp>
#include <memory>

#include "pcms/field/layout/mesh_fields.h"
#include "pcms/utility/types.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"
#include "pcms/localization/point_search.h"
#include "pcms/field/field.h" // OutOfBoundsMode

namespace pcms
{

// ---------------------------------------------------------------------------
// Abstract backend interface
// ---------------------------------------------------------------------------
template <typename T>
class MeshFieldBackend
{
public:
  virtual ~MeshFieldBackend() = default;
  virtual Kokkos::View<T* [1]> evaluate(Kokkos::View<T**> localCoords,
                                        Kokkos::View<LO*> offsets) const = 0;
  virtual void SetData(Rank1View<const T, HostMemorySpace> data,
                       size_t num_nodes, size_t num_components, int dim) = 0;
  virtual void GetData(Rank1View<T, HostMemorySpace> data, size_t num_nodes,
                       size_t num_components, int dim) const = 0;
};

// ---------------------------------------------------------------------------
// Concrete backend implementation
// ---------------------------------------------------------------------------
template <typename T, int Dim, int Order>
class MeshFieldBackendImpl : public MeshFieldBackend<T>
{
public:
  MeshFieldBackendImpl(Omega_h::Mesh& mesh)
    : mesh_(mesh),
      mesh_field_(mesh),
      shape_field_(mesh_field_.template CreateLagrangeField<T, Order, 1>())
  {
  }

  Kokkos::View<T* [1]> evaluate(Kokkos::View<T**> localCoords,
                                Kokkos::View<LO*> offsets) const override
  {
    auto self = const_cast<MeshFieldBackendImpl<T, Dim, Order>*>(this);
    return self->mesh_field_.triangleLocalPointEval(localCoords, offsets,
                                                    shape_field_);
  }

  void SetData(Rank1View<const T, HostMemorySpace> data, size_t num_nodes,
               size_t num_components, int dim) override
  {
    size_t stride = num_nodes * num_components;
    auto topo = static_cast<MeshField::Mesh_Topology>(dim);
    Kokkos::View<T*, DefaultExecutionSpace::memory_space> data_d("data_d",
                                                                 data.size());
    Kokkos::deep_copy(data_d, Kokkos::View<const T*, HostMemorySpace>(
                                data.data_handle(), data.size()));
    Kokkos::parallel_for(
      mesh_.nents(dim), KOKKOS_CLASS_LAMBDA(size_t ent) {
        for (size_t n = 0; n < num_nodes; ++n) {
          for (size_t c = 0; c < num_components; ++c) {
            shape_field_(ent, n, c, topo) =
              data_d[ent * stride + n * num_components + c];
          }
        }
      });
  }

  void GetData(Rank1View<T, HostMemorySpace> data, size_t num_nodes,
               size_t num_components, int dim) const override
  {
    size_t stride = num_nodes * num_components;
    auto topo = static_cast<MeshField::Mesh_Topology>(dim);
    Kokkos::View<T*, DefaultExecutionSpace::memory_space> data_d("data_d",
                                                                 data.size());
    Kokkos::parallel_for(
      mesh_.nents(dim), KOKKOS_CLASS_LAMBDA(size_t ent) {
        for (size_t n = 0; n < num_nodes; ++n) {
          for (size_t c = 0; c < num_components; ++c) {
            data_d[ent * stride + n * num_components + c] =
              shape_field_(ent, n, c, topo);
          }
        }
      });
    Kokkos::deep_copy(
      Kokkos::View<T*, HostMemorySpace>(data.data_handle(), data.size()),
      data_d);
  }

private:
  Omega_h::Mesh& mesh_;
  MeshField::OmegahMeshField<DefaultExecutionSpace, Dim> mesh_field_;
  using ShapeField =
    decltype(mesh_field_.template CreateLagrangeField<T, Order, 1>());
  ShapeField shape_field_;
};

// ---------------------------------------------------------------------------
// Factory function: create a MeshFieldBackend from a layout
// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<MeshFieldBackend<T>> MakeMeshFieldBackend(
  const MeshFieldsAdapterLayout& layout)
{
  Omega_h::Mesh& mesh = layout.GetMesh();
  if (mesh.dim() == 3) {
    throw pcms_error("MeshFieldBackend does not support 3D meshes");
  }
  auto nodes_per_dim = layout.GetNodesPerDim();
  if (nodes_per_dim[0] == 1 && nodes_per_dim[1] == 0 && nodes_per_dim[2] == 0 &&
      nodes_per_dim[3] == 0) {
    switch (mesh.dim()) {
      case 1: return std::make_shared<MeshFieldBackendImpl<T, 1, 1>>(mesh);
      case 2: return std::make_shared<MeshFieldBackendImpl<T, 2, 1>>(mesh);
      default: break;
    }
  } else if (nodes_per_dim[0] == 1 && nodes_per_dim[1] == 1 &&
             nodes_per_dim[2] == 0 && nodes_per_dim[3] == 0) {
    switch (mesh.dim()) {
      case 2: return std::make_shared<MeshFieldBackendImpl<T, 2, 2>>(mesh);
      case 3: return std::make_shared<MeshFieldBackendImpl<T, 3, 2>>(mesh);
      default: break;
    }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Helper functors used in MeshFieldsAdapter2LocalizationHint
// ---------------------------------------------------------------------------
struct ComputeOffsetsFunctor
{
  Kokkos::View<LO*, HostMemorySpace> offsets_;
  Kokkos::View<LO*, HostMemorySpace> elem_counts_;

  ComputeOffsetsFunctor(Kokkos::View<LO*, HostMemorySpace> offsets,
                        Kokkos::View<LO*, HostMemorySpace> elem_counts)
    : offsets_(offsets), elem_counts_(elem_counts)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(LO i, LO& partial, bool is_final) const
  {
    if (is_final) {
      offsets_(i) = partial;
    }
    partial += elem_counts_(i);
  }
};

struct CountPointsPerElementFunctor
{
  Kokkos::View<LO*> elem_counts_;
  Kokkos::View<GridPointSearch2D::Result*> search_results_;

  CountPointsPerElementFunctor(
    Kokkos::View<LO*> elem_counts,
    Kokkos::View<GridPointSearch2D::Result*> search_results)
    : elem_counts_(elem_counts), search_results_(search_results)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(LO i) const
  {
    auto [dim, elem_idx, coord] = search_results_(i);
    Kokkos::atomic_add(&elem_counts_(elem_idx), 1);
  }
};

struct FillCoordinatesAndIndicesFunctor
{
  Omega_h::Mesh& mesh_;
  Kokkos::View<LO*> elem_counts_;
  Kokkos::View<LO*> offsets_;
  Kokkos::View<Real**> coordinates_;
  Kokkos::View<LO*> indices_;
  Kokkos::View<GridPointSearch2D::Result*> search_results_;
  Omega_h::Int dim_;

  FillCoordinatesAndIndicesFunctor(
    Omega_h::Mesh& mesh, Kokkos::View<LO*> elem_counts,
    Kokkos::View<LO*> offsets, Kokkos::View<Real**> coordinates,
    Kokkos::View<LO*> indices,
    Kokkos::View<GridPointSearch2D::Result*> search_results)
    : mesh_(mesh),
      elem_counts_(elem_counts),
      offsets_(offsets),
      coordinates_(coordinates),
      indices_(indices),
      search_results_(search_results),
      dim_(mesh.dim())
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(LO i) const
  {
    auto [dim, elem_idx, coord] = search_results_(i);
    LO count = Kokkos::atomic_sub_fetch(&elem_counts_(elem_idx), 1);
    LO index = offsets_(elem_idx) + count - 1;
    for (int j = 0; j < (dim_ + 1); ++j) {
      coordinates_(index, j) = coord[j];
    }
    indices_(index) = i;
  }
};

// ---------------------------------------------------------------------------
// Localization hint for MeshFields
// ---------------------------------------------------------------------------
struct MeshFieldsAdapter2LocalizationHint
{
  MeshFieldsAdapter2LocalizationHint(
    Omega_h::Mesh& mesh,
    Kokkos::View<GridPointSearch2D::Result*, HostMemorySpace> search_results,
    OutOfBoundsMode mode)
    : mode_(mode), num_valid_(0), num_missing_(0)
  {
    std::vector<size_t> valid_point_indices;
    std::vector<size_t> missing_point_indices;

    if (mode_ == OutOfBoundsMode::ERROR) {
      for (size_t i = 0; i < search_results.size(); ++i) {
        auto [dim, elem_idx, coord] = search_results(i);
        bool is_missing =
          (static_cast<int>(dim) != mesh.dim()) || (elem_idx < 0);
        PCMS_ALWAYS_ASSERT(!is_missing && "Points found outside mesh domain");
        valid_point_indices.push_back(i);
      }
    } else {
      for (size_t i = 0; i < search_results.size(); ++i) {
        auto [dim, elem_idx, coord] = search_results(i);
        bool is_missing =
          (static_cast<int>(dim) != mesh.dim()) || (elem_idx < 0);
        if (is_missing) {
          missing_point_indices.push_back(i);
        } else {
          valid_point_indices.push_back(i);
        }
      }
    }

    num_valid_ = valid_point_indices.size();
    num_missing_ = missing_point_indices.size();

    if (num_missing_ > 0 && mode_ == OutOfBoundsMode::NEAREST_BOUNDARY) {
      PCMS_ALWAYS_ASSERT(false && "NEAREST_BOUNDARY mode not implemented yet");
    }

    offsets_ = Kokkos::View<LO*, HostMemorySpace>("offsets", mesh.nelems() + 1);
    coordinates_ = Kokkos::View<Real**, HostMemorySpace>(
      "coordinates", num_valid_, mesh.dim() + 1);
    indices_ = Kokkos::View<LO*, HostMemorySpace>("indices", num_valid_);

    if (num_missing_ > 0) {
      missing_indices_ =
        Kokkos::View<LO*, HostMemorySpace>("missing_indices", num_missing_);
      for (size_t i = 0; i < num_missing_; ++i) {
        missing_indices_(i) = static_cast<LO>(missing_point_indices[i]);
      }
    }

    Kokkos::View<LO*, HostMemorySpace> elem_counts("elem_counts",
                                                   mesh.nelems());
    for (size_t i = 0; i < num_valid_; ++i) {
      auto [dim, elem_idx, coord] = search_results(valid_point_indices[i]);
      elem_counts[elem_idx] += 1;
    }

    LO total;
    ComputeOffsetsFunctor functor(offsets_, elem_counts);
    Kokkos::parallel_scan(
      "ComputeOffsets",
      Kokkos::RangePolicy<HostMemorySpace::execution_space>(0, mesh.nelems()),
      functor, total);
    offsets_(mesh.nelems()) = total;

    for (size_t i = 0; i < num_valid_; ++i) {
      size_t orig_idx = valid_point_indices[i];
      auto [dim, elem_idx, coord] = search_results(orig_idx);
      elem_counts(elem_idx) -= 1;
      LO index = offsets_(elem_idx) + elem_counts(elem_idx);
      for (int j = 0; j < (mesh.dim() + 1); ++j) {
        coordinates_(index, j) = coord[j];
      }
      indices_(index) = static_cast<LO>(orig_idx);
    }
  }

  OutOfBoundsMode mode_;
  size_t num_valid_;
  size_t num_missing_;

  Kokkos::View<LO*, HostMemorySpace> offsets_;
  Kokkos::View<Real**, HostMemorySpace> coordinates_;
  Kokkos::View<LO*, HostMemorySpace> indices_;
  Kokkos::View<LO*, HostMemorySpace> missing_indices_;
};

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H
