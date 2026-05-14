#include <pcms/utility/eqdsk.h>
#include <pcms/utility/arrays.h>
#include <pcms/field/function_space/spline.h>
#include <pcms/field/coordinate_system.h>
#include <pcms/field/evaluation_request.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

using pcms::CoordinateSystem;
using pcms::EQDSKData;

// This test only verifies that the EQDSK data can be loaded and that the
// SplineFunctionSpace can be created and evaluated. The actual values of the
// loaded data and the spline evaluation are not checked.
int main(int argc, char** argv)
{
  auto lib = Omega_h::Library{};

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <eqdsk_file>\n";
    return 1;
  }

  const std::string filename(argv[1]);

  std::cout << "Loading EQDSK file: " << filename << "\n";

  try {
    // Test constructor from file
    EQDSKData eqdsk_data(filename);

    std::cout << "Grid dimensions: NW = " << eqdsk_data.GetNW()
              << ", NH = " << eqdsk_data.GetNH() << "\n";
    std::cout << "R range: [" << eqdsk_data.GetRMin() << ", "
              << eqdsk_data.GetRMax() << "]\n";
    std::cout << "Z range: [" << eqdsk_data.GetZMin() << ", "
              << eqdsk_data.GetZMax() << "]\n";
    std::cout << "Grid spacing: dR = " << eqdsk_data.GetDeltaR()
              << ", dZ = " << eqdsk_data.GetDeltaZ() << "\n";

    // Verify grid dimensions are positive
    assert(eqdsk_data.GetNW() > 0);
    assert(eqdsk_data.GetNH() > 0);

    // Verify grid bounds are valid
    assert(eqdsk_data.GetRMin() < eqdsk_data.GetRMax());
    assert(eqdsk_data.GetZMin() < eqdsk_data.GetZMax());

    // Verify grid dimensions match
    const double r_tol = 1e-10;
    const double z_tol = 1e-10;
    assert(std::abs(eqdsk_data.GetRDIM() -
                    (eqdsk_data.GetRMax() - eqdsk_data.GetRMin())) < r_tol);
    assert(std::abs(eqdsk_data.GetZDIM() -
                    (eqdsk_data.GetZMax() - eqdsk_data.GetZMin())) < z_tol);

    // Verify PSIZR array size
    const int expected_size = eqdsk_data.GetNW() * eqdsk_data.GetNH();
    assert(eqdsk_data.PSIZR.extent(0) == static_cast<size_t>(expected_size));

    assert(eqdsk_data.GetDeltaR() > 0.0);
    assert(eqdsk_data.GetDeltaZ() > 0.0);

    const int NW = eqdsk_data.GetNW();
    const int NH = eqdsk_data.GetNH();

    assert(eqdsk_data.GetPsiIndex(0, 0) == 0);
    assert(eqdsk_data.GetPsiIndex(NW - 1, 0) == NW - 1);
    assert(eqdsk_data.GetPsiIndex(0, NH - 1) == (NH - 1) * NW);
    assert(eqdsk_data.GetPsiIndex(NW - 1, NH - 1) == NW * NH - 1);

    const double coord_tol = 1e-10;
    assert(std::abs(eqdsk_data.GetR(0) - eqdsk_data.GetRMin()) < coord_tol);
    assert(std::abs(eqdsk_data.GetR(NW - 1) - eqdsk_data.GetRMax()) <
           coord_tol);
    assert(std::abs(eqdsk_data.GetZ(0) - eqdsk_data.GetZMin()) < coord_tol);
    assert(std::abs(eqdsk_data.GetZ(NH - 1) - eqdsk_data.GetZMax()) <
           coord_tol);

    std::cout << "\nTesting SplineFunctionSpace evaluation...\n";

    // Create a modified grid for spline interpolation
    // EQDSK grid.divisions = NW x NH (number of data points)
    // SplineFunctionSpace with order=1 needs grid.divisions = (NW-1) x (NH-1)
    // cells
    pcms::UniformGrid<2> spline_grid;
    spline_grid.bot_left = eqdsk_data.grid.bot_left;
    spline_grid.edge_length = eqdsk_data.grid.edge_length;
    spline_grid.divisions = {NW - 1, NH - 1}; // Number of cells, not points

    auto spline_space = pcms::SplineFunctionSpace::FromUniformGrid(
      spline_grid, CoordinateSystem::Cartesian);

    auto psi_field = spline_space.CreateField<pcms::Real>();

    auto layout = spline_space.GetLayout();
    const int layout_size = layout->OwnedSize();
    const int psizr_size = static_cast<int>(eqdsk_data.PSIZR.extent(0));
    std::cout << "Layout OwnedSize: " << layout_size << "\n";
    std::cout << "PSIZR size: " << psizr_size << " (NW=" << NW << " * NH=" << NH
              << ")\n";

    assert(layout_size == psizr_size);

    psi_field.GetData().SetDOFHolderData(
      pcms::Rank1View<const pcms::Real, pcms::DeviceMemorySpace>(
        eqdsk_data.PSIZR.data(), eqdsk_data.PSIZR.extent(0)));

    const pcms::Real R_mid =
      (eqdsk_data.GetRMin() + eqdsk_data.GetRMax()) / 2.0;
    const pcms::Real Z_mid =
      (eqdsk_data.GetZMin() + eqdsk_data.GetZMax()) / 2.0;

    std::vector<pcms::Real> eval_coords = {
      R_mid,
      Z_mid, // center
      eqdsk_data.GetRMin() + 0.25 * eqdsk_data.GetRDIM(),
      eqdsk_data.GetZMin() + 0.25 * eqdsk_data.GetZDIM(), // lower left
      eqdsk_data.GetRMin() + 0.75 * eqdsk_data.GetRDIM(),
      eqdsk_data.GetZMin() + 0.75 * eqdsk_data.GetZDIM() // upper right
    };

    const int num_eval_points = 3;

    auto eval_coords_host = Kokkos::View<pcms::Real**, pcms::HostMemorySpace>(
      "eval_coords_host", num_eval_points, 2);
    for (int i = 0; i < num_eval_points; ++i) {
      eval_coords_host(i, 0) = eval_coords[2 * i];     // R
      eval_coords_host(i, 1) = eval_coords[2 * i + 1]; // Z
    }

    auto eval_coords_device =
      Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace>("eval_coords_device",
                                                          num_eval_points, 2);
    pcms::DeepCopyMismatchLayouts(eval_coords_device, eval_coords_host);

    auto coords_view = pcms::MakeRank2View(eval_coords_device);
    auto coord_view = pcms::CoordinateView<pcms::DeviceMemorySpace>{
      CoordinateSystem::Cartesian, coords_view};
    auto eval_request = pcms::EvaluationRequest::FromCoordinates(coord_view);

    auto evaluator =
      spline_space.CreatePointEvaluator<pcms::Real>(eval_request);

    auto eval_results_1d = Kokkos::View<pcms::Real*, pcms::DeviceMemorySpace>(
      "eval_results", num_eval_points);

    using LayoutPolicy =
      pcms::detail::default_layout_for_memory_space_t<pcms::DeviceMemorySpace>;
    pcms::Rank2View<pcms::Real, pcms::DeviceMemorySpace, LayoutPolicy>
      eval_results(eval_results_1d.data(), num_eval_points, 1);

    evaluator->Evaluate(psi_field, eval_results);

    auto results_host = Kokkos::create_mirror_view_and_copy(
      pcms::HostMemorySpace(), eval_results_1d);

    std::cout << "Spline evaluation results:\n";
    for (int i = 0; i < num_eval_points; ++i) {
      std::cout << "  Point " << i << ": (R=" << eval_coords[2 * i]
                << ", Z=" << eval_coords[2 * i + 1]
                << ") -> Psi = " << results_host(i) << "\n";
    }

    for (int i = 0; i < num_eval_points; ++i) {
      assert(std::isfinite(results_host(i)));
    }

    std::cout << "SplineFunctionSpace evaluation test passed!\n";

    std::cout << "\nAll tests passed!\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
