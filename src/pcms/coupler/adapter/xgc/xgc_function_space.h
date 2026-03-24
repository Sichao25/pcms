#ifndef PCMS_XGC_FUNCTION_SPACE_H
#define PCMS_XGC_FUNCTION_SPACE_H

#include "pcms/coupler/adapter/xgc/xgc_field_data.h"
#include "pcms/field/field.h"
#include "pcms/field/function_space.h"
#include "pcms/field/field_metadata.h"
#include "pcms/utility/common.h"
#include <functional>
#include <memory>

namespace pcms
{

class XGCFunctionSpace : public FunctionSpace
{
public:
  XGCFunctionSpace(const ReverseClassificationVertex& reverse_classification,
                   std::function<int8_t(int, int)> in_overlap,
                   LO num_plane_nodes)
    : layout_(std::make_shared<XGCFieldLayout>(reverse_classification,
                                               std::move(in_overlap),
                                               num_plane_nodes))
  {
  }

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout() const noexcept override
  {
    return layout_;
  }

  [[nodiscard]] CoordinateSystem GetCoordinateSystem() const noexcept override
  {
    return CoordinateSystem::XGC;
  }

  [[nodiscard]] const FieldEvaluatorFactory<Real>& GetEvaluatorFactory() const override
  {
    throw pcms_error("XGCFunctionSpace does not support evaluator factories");
  }

  template <typename T>
  [[nodiscard]] Field<T> CreateField(Rank1View<T, HostMemorySpace> data,
                                     FieldMetadata metadata = {}) const
  {
    return Field<T>(nullptr,
                    std::make_unique<XGCFieldData<T>>(layout_, metadata, data));
  }

private:
  std::shared_ptr<const XGCFieldLayout> layout_;
};

} // namespace pcms

#endif // PCMS_XGC_FUNCTION_SPACE_H
