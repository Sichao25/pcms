#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pcms/field/nodal_field_factory.h"
#include "pcms/field/adapter/point_cloud/point_cloud_layout.h"
#include "field_test_utils.h"

#include <vector>

using pcms::CoordinateSystem;
using pcms::HostMemorySpace;
using pcms::LO;
using pcms::Rank1View;
using pcms::Rank2View;
using pcms::Real;

namespace
{

std::vector<Real> MakeCoords2D()
{
  return {
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0
  };
}

} // namespace

TEST_CASE("NodalFieldFactory creates point-cloud layout metadata")
{
  auto coords = MakeCoords2D();
  Rank2View<Real, HostMemorySpace> coords_view(coords.data(), 4, 2);

  auto factory =
    pcms::NodalFieldFactory::Create(coords_view, CoordinateSystem::Cartesian);
  auto layout = factory.GetLayout();

  REQUIRE(layout->GetNumComponents() == 1);
  REQUIRE(layout->GetNumOwnedDofHolder() == 4);
  REQUIRE(layout->GetNumGlobalDofHolder() == 4);
  REQUIRE_FALSE(layout->IsDistributed());

  auto dof_coords = layout->GetDOFHolderCoordinates().GetCoordinates();
  REQUIRE(static_cast<int>(dof_coords.extent(0)) == 4);
  REQUIRE(static_cast<int>(dof_coords.extent(1)) == 2);
  for (int i = 0; i < 4; ++i) {
    REQUIRE(dof_coords(i, 0) == Catch::Approx(coords[2 * i]));
    REQUIRE(dof_coords(i, 1) == Catch::Approx(coords[2 * i + 1]));
  }
}

TEST_CASE("NodalFieldFactory fields share layout")
{
  auto coords = MakeCoords2D();
  Rank2View<Real, HostMemorySpace> coords_view(coords.data(), 4, 2);

  auto factory =
    pcms::NodalFieldFactory::Create(coords_view, CoordinateSystem::Cartesian);
  auto source = factory.CreateFieldReal();
  auto target = factory.CreateFieldReal();

  REQUIRE(&source->GetLayout() == &target->GetLayout());
}

TEST_CASE("NodalFieldFactory point-cloud field set/get DOF round-trip")
{
  auto coords = MakeCoords2D();
  Rank2View<Real, HostMemorySpace> coords_view(coords.data(), 4, 2);

  auto field =
    pcms::NodalFieldFactory::Create(coords_view, CoordinateSystem::Cartesian)
      .CreateFieldReal();

  std::vector<Real> data{1.0, 2.0, 3.0, 4.0};
  Rank1View<const Real, HostMemorySpace> data_view(data.data(), data.size());
  field->SetDOFHolderData(data_view);

  auto got = field->GetDOFHolderData();
  REQUIRE(got.size() == data.size());
  for (LO i = 0; i < static_cast<LO>(data.size()); ++i) {
    REQUIRE(got[i] == Catch::Approx(data[i]));
  }
}

TEST_CASE("NodalFieldFactory point-cloud field serialize / deserialize round-trip")
{
  auto coords = MakeCoords2D();
  Rank2View<Real, HostMemorySpace> coords_view(coords.data(), 4, 2);

  auto field =
    pcms::NodalFieldFactory::Create(coords_view, CoordinateSystem::Cartesian)
      .CreateFieldReal();

  std::vector<Real> data{5.0, 6.0, 7.0, 8.0};
  Rank1View<const Real, HostMemorySpace> data_view(data.data(), data.size());
  field->SetDOFHolderData(data_view);

  pcms::test::CheckSerializeDeserialize(*field);
}

TEST_CASE("NodalFieldFactory field keeps layout alive after temporary factory destruction")
{
  auto coords = MakeCoords2D();
  Rank2View<Real, HostMemorySpace> coords_view(coords.data(), 4, 2);

  auto field =
    pcms::NodalFieldFactory::Create(coords_view, CoordinateSystem::Cartesian)
      .CreateFieldReal();

  auto point_cloud_layout =
    dynamic_cast<const pcms::PointCloudLayout*>(&field->GetLayout());
  REQUIRE(point_cloud_layout != nullptr);
  REQUIRE(point_cloud_layout->GetNumOwnedDofHolder() == 4);

  std::vector<Real> data{9.0, 10.0, 11.0, 12.0};
  Rank1View<const Real, HostMemorySpace> data_view(data.data(), data.size());
  field->SetDOFHolderData(data_view);

  auto got = field->GetDOFHolderData();
  REQUIRE(got[0] == Catch::Approx(9.0));
  REQUIRE(got[3] == Catch::Approx(12.0));
}
