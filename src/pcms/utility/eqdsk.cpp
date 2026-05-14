#include "pcms/utility/eqdsk.h"
#include "pcms/utility/assert.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cctype>

namespace pcms
{

void readgfile(const std::string& filename, GEQDData& eqd)
{
  constexpr double pi = 3.1415926535897932;
  constexpr double tmu = 2.0e-07;
  constexpr double twopi = 2.0 * pi;

  std::ifstream fin(filename);

  if (!fin) {
    throw std::runtime_error("Cannot open file");
  }

  // ----------------------------------------
  // Read header
  // ----------------------------------------
  // G-EQDSK format (Fortran format: 6a8,3i4):
  //   6 strings of 8 characters each (characters 1-48)
  //   3 integers of 4 characters each (characters 49-60)
  // Example: "  EFITD    05/12/98      # 96333 ,3337ms           3  65  65"

  std::string first_line;
  std::getline(fin, first_line);

  // Ensure line is long enough (at least 60 characters for full format)
  if (first_line.length() < 60) {
    first_line.resize(60, ' ');
  }

  // Read 6 strings of 8 characters each (positions 0-47)
  for (int i = 0; i < 6; i++) {
    eqd.eqd_case[i] = first_line.substr(i * 8, 8);
  }

  // Read 3 integers of 4 characters each (positions 48-59)
  std::string imfit_str = first_line.substr(48, 4);
  std::string mw_str = first_line.substr(52, 4);
  std::string mh_str = first_line.substr(56, 4);

  eqd.eqd_imfit = std::stoi(imfit_str);
  eqd.eqd_mw = std::stoi(mw_str);
  eqd.eqd_mh = std::stoi(mh_str);

  // ----------------------------------------
  // Allocate arrays
  // ----------------------------------------

  int mw = eqd.eqd_mw;
  int mh = eqd.eqd_mh;

  eqd.eqd_fpol.resize(mw);
  eqd.eqd_pres.resize(mw);
  eqd.eqd_qpsi.resize(mw);
  eqd.eqd_workk.resize(mw);
  eqd.eqd_ffprim.resize(mw);
  eqd.eqd_pprime.resize(mw);
  eqd.eqd_xsi.resize(mw);

  eqd.eqd_rgrid.resize(mw);
  eqd.eqd_zgrid.resize(mh);
  eqd.eqd_psi_grid.resize(mw);

  eqd.eqd_psirz.resize(mw, std::vector<double>(mh));

  // ----------------------------------------
  // Read geometry
  // ----------------------------------------

  fin >> eqd.eqd_rdim >> eqd.eqd_zdim >> eqd.eqd_rzero >> eqd.eqd_r_min >>
    eqd.eqd_zmid;

  fin >> eqd.eqd_rmaxis >> eqd.eqd_zmaxis >> eqd.eqd_ssimag >> eqd.eqd_ssibry >>
    eqd.eqd_bzero;

  double xdum;
  double sdum;

  fin >> eqd.eqd_cpasma >> xdum >> xdum >> eqd.eqd_rmaxis >> xdum;

  fin >> eqd.eqd_zmaxis >> xdum >> sdum >> xdum >> xdum;

  // ----------------------------------------
  // Read profiles
  // ----------------------------------------

  for (int i = 0; i < mw; i++) {
    fin >> eqd.eqd_fpol[i];
  }

  for (int i = 0; i < mw; i++) {
    fin >> eqd.eqd_pres[i];
  }

  for (int i = 0; i < mw; i++) {
    fin >> eqd.eqd_workk[i];
  }

  eqd.eqd_drgrid = eqd.eqd_rdim / double(mw - 1);
  eqd.eqd_dzgrid = eqd.eqd_zdim / double(mh - 1);

  eqd.eqd_z_min = eqd.eqd_zmid - eqd.eqd_zdim / 2.0;

  eqd.eqd_darea = eqd.eqd_drgrid * eqd.eqd_dzgrid;

  for (int i = 0; i < mw; i++) {
    if (eqd.eqd_imfit >= 0) {
      eqd.eqd_ffprim[i] = -eqd.eqd_workk[i] / (twopi * tmu);
    } else {
      eqd.eqd_ffprim[i] = -eqd.eqd_workk[i];
    }
  }

  // ----------------------------------------
  // pprime
  // ----------------------------------------

  for (int i = 0; i < mw; i++) {
    fin >> eqd.eqd_workk[i];
  }

  for (int i = 0; i < mw; i++) {
    eqd.eqd_pprime[i] = -eqd.eqd_workk[i];
  }

  // ----------------------------------------
  // Read psirz
  // ----------------------------------------

  for (int j = 0; j < mh; j++) {
    for (int i = 0; i < mw; i++) {
      fin >> eqd.eqd_psirz[i][j];
    }
  }

  // ----------------------------------------
  // qpsi
  // ----------------------------------------

  for (int i = 0; i < mw; i++) {
    fin >> eqd.eqd_qpsi[i];
  }

  // ----------------------------------------
  // Boundary info
  // ----------------------------------------

  fin >> eqd.eqd_nbdry >> eqd.eqd_nlim;

  eqd.eqd_rbdry.resize(eqd.eqd_nbdry);
  eqd.eqd_zbdry.resize(eqd.eqd_nbdry);

  eqd.eqd_rlim.resize(eqd.eqd_nlim);
  eqd.eqd_zlim.resize(eqd.eqd_nlim);

  for (int i = 0; i < eqd.eqd_nbdry; i++) {
    fin >> eqd.eqd_rbdry[i] >> eqd.eqd_zbdry[i];
  }

  for (int i = 0; i < eqd.eqd_nlim; i++) {
    fin >> eqd.eqd_rlim[i] >> eqd.eqd_zlim[i];
  }

  fin.close();

  // ----------------------------------------
  // Apply psi factor
  // ----------------------------------------

  for (int i = 0; i < mw; i++) {
    for (int j = 0; j < mh; j++) {
      eqd.eqd_psirz[i][j] *= eqd.eqd_psi_factor;
    }
  }

  eqd.eqd_ssimag *= eqd.eqd_psi_factor;
  eqd.eqd_ssibry *= eqd.eqd_psi_factor;

  // ----------------------------------------
  // Build grids
  // ----------------------------------------

  eqd.eqd_r_max = eqd.eqd_r_min + eqd.eqd_rdim;
  eqd.eqd_z_max = eqd.eqd_z_min + eqd.eqd_zdim;

  for (int i = 0; i < mw; i++) {
    eqd.eqd_rgrid[i] = eqd.eqd_r_min + (eqd.eqd_r_max - eqd.eqd_r_min) *
                                         double(i) / double(mw - 1);
  }

  for (int i = 0; i < mh; i++) {
    eqd.eqd_zgrid[i] = eqd.eqd_z_min + (eqd.eqd_z_max - eqd.eqd_z_min) *
                                         double(i) / double(mh - 1);
  }

  for (int i = 0; i < mw; i++) {
    eqd.eqd_psi_grid[i] =
      double(i) / double(mw - 1) * (eqd.eqd_ssibry - eqd.eqd_ssimag) +
      eqd.eqd_ssimag;
  }

  std::cout << "Finished reading G-EQDSK file\n";
}

void readeqdfile(const std::string& filename, EQDData& eqd)
{
  std::ifstream fin(filename);

  if (!fin) {
    throw std::runtime_error("Cannot open eqd file");
  }

  // ----------------------------------------
  // Read header
  // ----------------------------------------

  std::string eq_header;
  std::getline(fin, eq_header);

  // ----------------------------------------
  // Read dimensions
  // ----------------------------------------

  fin >> eqd.eqd_mw >> eqd.eqd_mh >> eqd.eqd_mpsi;

  int mw = eqd.eqd_mw;
  int mh = eqd.eqd_mh;
  int mpsi = eqd.eqd_mpsi;

  // ----------------------------------------
  // Allocate arrays
  // ----------------------------------------

  eqd.eqd_psi_grid.resize(mpsi);

  eqd.eq_I.resize(mpsi);

  eqd.eqd_rgrid.resize(mw);
  eqd.eqd_zgrid.resize(mh);

  eqd.eqd_psirz.resize(mw, std::vector<double>(mh));

  // ----------------------------------------
  // Read geometry
  // ----------------------------------------

  fin >> eqd.eqd_r_min >> eqd.eqd_r_max >> eqd.eqd_z_min >> eqd.eqd_z_max;

  double eq_axis_b;

  fin >> eqd.eqd_rmaxis >> eqd.eqd_zmaxis >> eq_axis_b;

  // dummy variables
  double eq_x_psi_loc;
  double eq_x_r;
  double eq_x_z;

  fin >> eq_x_psi_loc >> eq_x_r >> eq_x_z;

  // ----------------------------------------
  // Read psi grid
  // ----------------------------------------

  for (int i = 0; i < mpsi; i++) {
    fin >> eqd.eqd_psi_grid[i];
  }

  std::cout << "Axis (R,Z,B) = " << eqd.eqd_rmaxis << " " << eqd.eqd_zmaxis
            << " " << eq_axis_b << "\n";

  // ----------------------------------------
  // Read I(psi)
  // ----------------------------------------

  for (int i = 0; i < mpsi; i++) {
    fin >> eqd.eq_I[i];
  }

  // ----------------------------------------
  // Read psi(R,Z)
  // ----------------------------------------

  // Keep same indexing convention
  // as original Fortran:
  //
  // eqd_psirz(i,j)
  //
  for (int j = 0; j < mh; j++) {
    for (int i = 0; i < mw; i++) {
      fin >> eqd.eqd_psirz[i][j];
    }
  }

  // ----------------------------------------
  // End flag
  // ----------------------------------------

  int end_flag;

  fin >> end_flag;

  if (end_flag != -1) {
    throw std::runtime_error("Wrong EQD file format");
  }

  fin.close();

  // ----------------------------------------
  // Build R grid
  // ----------------------------------------

  for (int i = 0; i < mw; i++) {

    eqd.eqd_rgrid[i] = eqd.eqd_r_min + (eqd.eqd_r_max - eqd.eqd_r_min) *
                                         double(i) / double(mw - 1);
  }

  // ----------------------------------------
  // Build Z grid
  // ----------------------------------------

  for (int i = 0; i < mh; i++) {

    eqd.eqd_zgrid[i] = eqd.eqd_z_min + (eqd.eqd_z_max - eqd.eqd_z_min) *
                                         double(i) / double(mh - 1);
  }

  // ----------------------------------------
  // Construct rectangular limiter
  // ----------------------------------------

  eqd.eqd_nlim = 4;

  eqd.eqd_rlim.resize(4);
  eqd.eqd_zlim.resize(4);

  eqd.eqd_rlim[0] = eqd.eqd_rgrid[0];
  eqd.eqd_zlim[0] = eqd.eqd_zgrid[0];

  eqd.eqd_rlim[1] = eqd.eqd_rgrid[0];
  eqd.eqd_zlim[1] = eqd.eqd_zgrid[mh - 1];

  eqd.eqd_rlim[2] = eqd.eqd_rgrid[mw - 1];
  eqd.eqd_zlim[2] = eqd.eqd_zgrid[mh - 1];

  eqd.eqd_rlim[3] = eqd.eqd_rgrid[mw - 1];
  eqd.eqd_zlim[3] = eqd.eqd_zgrid[0];

  std::cout << "EQD file loaded successfully\n";
}

// ============================================================================
// EQDSKData Constructors
// ============================================================================

// Constructor from filename
EQDSKData::EQDSKData(const std::string& filename)
{
  // Determine file type by extension
  std::string extension;
  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos != std::string::npos) {
    extension = filename.substr(dot_pos);
    // Convert to lowercase for case-insensitive comparison
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  }

  // Check for EQD format extensions
  if (extension == ".eqd") {
    EQDData eqd;
    readeqdfile(filename, eqd);
    *this = EQDSKData(eqd);
  }
  // Default to G-EQDSK format for .geq, .geqdsk, .g, or any other extension
  else {
    GEQDData geqd;
    readgfile(filename, geqd);
    *this = EQDSKData(geqd);
  }
}

// Constructor from GEQDData (G-EQDSK format)
EQDSKData::EQDSKData(const GEQDData& geqd)
{
  // Initialize grid structure
  grid.divisions[0] = static_cast<LO>(geqd.eqd_mw);
  grid.divisions[1] = static_cast<LO>(geqd.eqd_mh);
  grid.edge_length[0] = static_cast<Real>(geqd.eqd_rdim);
  grid.edge_length[1] = static_cast<Real>(geqd.eqd_zdim);
  grid.bot_left[0] = static_cast<Real>(geqd.eqd_r_min);
  grid.bot_left[1] = static_cast<Real>(geqd.eqd_z_min);

  // Allocate and copy 2D PSIZR array (column-major: R varies fastest)
  const int nw = geqd.eqd_mw;
  const int nh = geqd.eqd_mh;
  PSIZR = Kokkos::View<Real*, DeviceMemorySpace>("PSIZR", nw * nh);
  auto psizr_host = Kokkos::create_mirror_view(PSIZR);
  for (int j = 0; j < nh; ++j) {
    for (int i = 0; i < nw; ++i) {
      psizr_host(i + j * nw) = static_cast<Real>(geqd.eqd_psirz[i][j]);
    }
  }
  Kokkos::deep_copy(PSIZR, psizr_host);
}

// Constructor from EQDData (EQD format)
EQDSKData::EQDSKData(const EQDData& eqd)
{
  // Initialize grid structure
  grid.divisions[0] = static_cast<LO>(eqd.eqd_mw);
  grid.divisions[1] = static_cast<LO>(eqd.eqd_mh);
  grid.edge_length[0] = static_cast<Real>(eqd.eqd_r_max - eqd.eqd_r_min);
  grid.edge_length[1] = static_cast<Real>(eqd.eqd_z_max - eqd.eqd_z_min);
  grid.bot_left[0] = static_cast<Real>(eqd.eqd_r_min);
  grid.bot_left[1] = static_cast<Real>(eqd.eqd_z_min);

  // Allocate and copy 2D PSIZR array (column-major: R varies fastest)
  const int nw = eqd.eqd_mw;
  const int nh = eqd.eqd_mh;
  PSIZR = Kokkos::View<Real*, DeviceMemorySpace>("PSIZR", nw * nh);
  auto psizr_host = Kokkos::create_mirror_view(PSIZR);
  for (int j = 0; j < nh; ++j) {
    for (int i = 0; i < nw; ++i) {
      psizr_host(i + j * nw) = static_cast<Real>(eqd.eqd_psirz[i][j]);
    }
  }
  Kokkos::deep_copy(PSIZR, psizr_host);
}

} // namespace pcms
