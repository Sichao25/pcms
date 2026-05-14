#ifndef PCMS_UTILITY_EQDSK_H
#define PCMS_UTILITY_EQDSK_H

#include "pcms/utility/types.h"
#include "pcms/utility/uniform_grid.h"
#include "pcms/utility/memory_spaces.h"
#include <vector>
#include <string>
#include <array>

namespace pcms
{

struct GEQDData
{
  // Header
  std::array<std::string, 6> eqd_case;

  int eqd_imfit;
  int eqd_mw;
  int eqd_mh;

  // Scalars
  double eqd_rdim;
  double eqd_zdim;
  double eqd_rzero;
  double eqd_r_min;
  double eqd_zmid;

  double eqd_rmaxis;
  double eqd_zmaxis;
  double eqd_ssimag;
  double eqd_ssibry;
  double eqd_bzero;

  double eqd_cpasma;

  double eqd_drgrid;
  double eqd_dzgrid;
  double eqd_z_min;
  double eqd_r_max;
  double eqd_z_max;
  double eqd_darea;

  int eqd_nbdry;
  int eqd_nlim;

  double eqd_psi_factor = 1.0;

  // 1D arrays
  std::vector<double> eqd_fpol;
  std::vector<double> eqd_pres;
  std::vector<double> eqd_qpsi;
  std::vector<double> eqd_workk;
  std::vector<double> eqd_ffprim;
  std::vector<double> eqd_pprime;
  std::vector<double> eqd_xsi;

  std::vector<double> eqd_rgrid;
  std::vector<double> eqd_zgrid;
  std::vector<double> eqd_psi_grid;

  std::vector<double> eqd_rbdry;
  std::vector<double> eqd_zbdry;

  std::vector<double> eqd_rlim;
  std::vector<double> eqd_zlim;

  // 2D array
  std::vector<std::vector<double>> eqd_psirz;
};

struct EQDData
{
  int eqd_mw;
  int eqd_mh;
  int eqd_mpsi;

  int eqd_nlim;

  double eqd_r_min;
  double eqd_r_max;
  double eqd_z_min;
  double eqd_z_max;

  double eqd_rmaxis;
  double eqd_zmaxis;

  std::vector<double> eqd_psi_grid;
  std::vector<double> eq_I;

  std::vector<double> eqd_rgrid;
  std::vector<double> eqd_zgrid;

  std::vector<double> eqd_rlim;
  std::vector<double> eqd_zlim;

  // 2D psi array
  std::vector<std::vector<double>> eqd_psirz;
};

/**
 * @brief Data structure for EQDSK equilibrium data (simplified for field
 * evaluation)
 *
 * Contains only the essential data needed for poloidal flux field evaluation:
 * - Uniform 2D grid structure (R-Z computational domain)
 * - PSIZR: Poloidal flux values on grid points
 *
 * All grid data is stored in column-major order (R varies fastest).
 */
struct EQDSKData
{
  // Grid structure for the R-Z computational domain
  // grid.divisions[0] = NW (number of R grid points)
  // grid.divisions[1] = NH (number of Z grid points)
  // grid.edge_length[0] = RDIM (horizontal dimension in meter)
  // grid.edge_length[1] = ZDIM (vertical dimension in meter)
  // grid.bot_left[0] = RLEFT (minimum R in meter)
  // grid.bot_left[1] = Z_min (minimum Z in meter)
  Uniform2DGrid grid;

  // PSIZR: Poloidal flux in Weber/rad on the rectangular grid points
  Kokkos::View<Real*, DeviceMemorySpace> PSIZR;

  // Constructors
  // Constructor from file
  EQDSKData(const std::string& filename);

  // Constructor from GEQDData (G-EQDSK format)
  explicit EQDSKData(const GEQDData& geqd);

  // Constructor from EQDData (EQD format)
  explicit EQDSKData(const EQDData& eqd);

  // Accessor methods for backward compatibility
  [[nodiscard]] int GetNW() const noexcept { return grid.divisions[0]; }
  [[nodiscard]] int GetNH() const noexcept { return grid.divisions[1]; }
  [[nodiscard]] Real GetRDIM() const noexcept { return grid.edge_length[0]; }
  [[nodiscard]] Real GetZDIM() const noexcept { return grid.edge_length[1]; }
  [[nodiscard]] Real GetRLEFT() const noexcept { return grid.bot_left[0]; }
  [[nodiscard]] Real GetZMID() const noexcept
  {
    return grid.bot_left[1] + grid.edge_length[1] / 2.0;
  }

  // Grid bounds
  [[nodiscard]] Real GetZMin() const noexcept { return grid.bot_left[1]; }

  [[nodiscard]] Real GetZMax() const noexcept
  {
    return grid.bot_left[1] + grid.edge_length[1];
  }

  [[nodiscard]] Real GetRMin() const noexcept { return grid.bot_left[0]; }

  [[nodiscard]] Real GetRMax() const noexcept
  {
    return grid.bot_left[0] + grid.edge_length[0];
  }

  [[nodiscard]] Real GetDeltaR() const noexcept
  {
    const int NW = grid.divisions[0];
    return (NW > 1) ? grid.edge_length[0] / static_cast<Real>(NW - 1) : 0.0;
  }

  [[nodiscard]] Real GetDeltaZ() const noexcept
  {
    const int NH = grid.divisions[1];
    return (NH > 1) ? grid.edge_length[1] / static_cast<Real>(NH - 1) : 0.0;
  }

  [[nodiscard]] int GetPsiIndex(int i_R, int i_Z) const noexcept
  {
    return i_R + i_Z * grid.divisions[0];
  }

  [[nodiscard]] Real GetPsi(int i_R, int i_Z) const
  {
    return PSIZR[GetPsiIndex(i_R, i_Z)];
  }

  [[nodiscard]] Real GetR(int i_R) const noexcept
  {
    return grid.bot_left[0] + i_R * GetDeltaR();
  }

  [[nodiscard]] Real GetZ(int i_Z) const noexcept
  {
    return grid.bot_left[1] + i_Z * GetDeltaZ();
  }
};

} // namespace pcms

#endif // PCMS_UTILITY_EQDSK_H
