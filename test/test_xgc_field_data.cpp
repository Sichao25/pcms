#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "pcms/coupler/adapter/xgc/xgc_field_data.h"
#include "pcms/coupler/adapter/xgc/xgc_field_serializer.h"
#include <algorithm>
#include <numeric>

namespace
{

pcms::ReverseClassificationVertex create_dummy_rc(int size)
{
  pcms::ReverseClassificationVertex rc;
  for (int i = 0; i < size; ++i) {
    if (i % 4 == 0) {
      rc.Insert({0, 0}, i);
    } else {
      rc.Insert({0, 1}, i);
    }
  }
  return rc;
}

bool in_overlap(int, int id)
{
  return id == 0;
}

} // namespace

TEST_CASE("XGC FieldLayout marks overlap entries and gids")
{
  static constexpr int data_size = 16;
  auto rc = create_dummy_rc(data_size);
  pcms::XGCFieldLayout layout(rc, in_overlap, data_size);

  auto owned = layout.GetOwned();
  auto gids = layout.GetGids();
  auto class_dims = layout.GetDOFHolderClassificationDimensions();
  auto class_ids = layout.GetDOFHolderClassificationIds();

  for (int i = 0; i < data_size; ++i) {
    REQUIRE(gids[i] == i + 1);
    if (i % 4 == 0) {
      REQUIRE(owned[i]);
      REQUIRE(class_dims[i] == 0);
      REQUIRE(class_ids[i] == 0);
    } else {
      REQUIRE(!owned[i]);
      REQUIRE(class_dims[i] == -1);
      REQUIRE(class_ids[i] == -1);
    }
  }
}

TEST_CASE("XGC FieldData serializer preserves inactive entries")
{
  static constexpr int data_size = 16;
  auto rc = create_dummy_rc(data_size);
  auto layout = std::make_shared<pcms::XGCFieldLayout>(rc, in_overlap, data_size);

  std::vector<pcms::Real> data(data_size);
  std::iota(data.begin(), data.end(), 0.0);
  auto original = data;
  pcms::XGCFieldData<pcms::Real> field(
    layout, pcms::FieldMetadata{}, pcms::make_array_view(data));
  pcms::XGCFieldSerializer<pcms::Real> serializer(MPI_COMM_SELF);

  std::vector<pcms::LO> permutation(data_size);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::vector<pcms::Real> buffer(data_size, -1.0);

  auto owned = layout->GetOwned();
  const int num_owned =
    std::count_if(owned.data_handle(), owned.data_handle() + data_size,
                  [](bool is_owned) { return is_owned; });

  REQUIRE(serializer.Serialize(field, pcms::make_array_view(buffer),
                               pcms::make_const_array_view(permutation)) ==
          data_size);
  REQUIRE(num_owned == 4);

  for (int i = 0; i < data_size; ++i) {
    if (owned[i]) {
      REQUIRE(buffer[i] == data[i]);
      buffer[i] += 100.0;
    } else {
      REQUIRE(buffer[i] == -1.0);
    }
  }

  serializer.Deserialize(field, pcms::make_const_array_view(buffer),
                         pcms::make_const_array_view(permutation));

  auto after = field.GetDOFHolderDataHost();
  for (int i = 0; i < data_size; ++i) {
    if (owned[i]) {
      REQUIRE(after[i] == Catch::Approx(original[i] + 100.0));
    } else {
      REQUIRE(after[i] == Catch::Approx(original[i]));
    }
  }
}
