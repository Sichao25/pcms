#ifndef PCMS_LOCALIZATION_PATH_SELECTION_H
#define PCMS_LOCALIZATION_PATH_SELECTION_H

#include "pcms/field/field_layout.h"
#include "pcms/utility/entity_types.h"

#include <optional>

namespace pcms::detail
{

enum class LocalizationPath
{
  PointCloudSupports,
  VertexAdjacencySearch,
  CentroidToVertexAdjacencySearch
};

// Returns the unique entity dimension represented by all DOF holders in the
// layout when it maps one-to-one onto a discretization entity set. Otherwise,
// returns std::nullopt.
inline std::optional<int> GetUniformDofHolderEntityDim(
  const FieldLayout& layout)
{
  auto disc = layout.GetDiscretization();
  if (disc == nullptr) {
    return std::nullopt;
  }

  auto class_dims = layout.GetDOFHolderClassificationDimensions();
  if (class_dims.extent(0) == 0) {
    return std::nullopt;
  }

  const int entity_dim = static_cast<int>(class_dims(0));
  for (decltype(class_dims.extent(0)) i = 1; i < class_dims.extent(0); ++i) {
    if (class_dims(i) != entity_dim) {
      return std::nullopt;
    }
  }

  if (layout.GetNumOwnedDofHolder() != disc->GetNumEntities(entity_dim)) {
    return std::nullopt;
  }

  return entity_dim;
}

inline LocalizationPath SelectLocalizationPath(
  const FieldLayout& source_layout,
  const FieldLayout* target_layout = nullptr)
{
  auto source_disc = source_layout.GetDiscretization();
  if (source_disc == nullptr) {
    return LocalizationPath::PointCloudSupports;
  }

  auto source_entity_dim = GetUniformDofHolderEntityDim(source_layout);
  if (source_entity_dim.has_value() && *source_entity_dim == Vertex) {
    return LocalizationPath::VertexAdjacencySearch;
  }

  if (target_layout == nullptr) {
    return LocalizationPath::PointCloudSupports;
  }

  auto target_disc = target_layout->GetDiscretization();
  auto target_entity_dim = GetUniformDofHolderEntityDim(*target_layout);
  if (source_entity_dim.has_value() &&
      *source_entity_dim == Face &&
      target_entity_dim.has_value() &&
      *target_entity_dim == Vertex &&
      target_disc != nullptr &&
      source_disc->GetDimension() == 2 &&
      source_disc->SameEntities(*target_disc)) {
    return LocalizationPath::CentroidToVertexAdjacencySearch;
  }

  return LocalizationPath::PointCloudSupports;
}

} // namespace pcms::detail

#endif // PCMS_LOCALIZATION_PATH_SELECTION_H
