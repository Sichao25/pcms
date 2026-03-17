#ifndef PCMS_LAGRANGE_FIELD_FACTORY_H
#define PCMS_LAGRANGE_FIELD_FACTORY_H

#include <Omega_h_mesh.hpp>
#include "pcms/field.h"
#include "pcms/field_layout.h"
#include "pcms/coordinate_system.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/types.h"
#include "pcms/utility/memory_spaces.h"

#include <functional>
#include <memory>
#include <string>

namespace pcms
{

class LagrangeFieldFactory
{
public:
  // Unstructured mesh — delegates to MeshFieldsAdapterLayout
  [[nodiscard]] static LagrangeFieldFactory FromMesh(
    Omega_h::Mesh& mesh, int order, int num_components,
    CoordinateSystem coordinate_system,
    std::string global_id_name = "global");

  // Structured uniform grid — order-1 H1-conforming nodal field on a regular grid
  [[nodiscard]] static LagrangeFieldFactory FromUniformGrid(
    Rank1View<Real, HostMemorySpace> edge_length,
    Rank1View<Real, HostMemorySpace> bot_left,
    Rank1View<LO, HostMemorySpace>   divisions,
    int num_components,
    CoordinateSystem coordinate_system);

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout() const noexcept;
  [[nodiscard]] std::unique_ptr<FieldT<Real>> CreateFieldReal() const;

private:
  explicit LagrangeFieldFactory(
    std::shared_ptr<const FieldLayout> layout,
    std::function<std::unique_ptr<FieldT<Real>>()> create_fn) noexcept;

  std::shared_ptr<const FieldLayout> layout_;
  std::function<std::unique_ptr<FieldT<Real>>()> create_fn_;
};

} // namespace pcms

#endif // PCMS_LAGRANGE_FIELD_FACTORY_H
