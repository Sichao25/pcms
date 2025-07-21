#include "mdspan/mdspan.hpp"
#include "pcms.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <pcms/interpolator/spline_interpolator.hpp>

using namespace pcms;

bool are_equal(double a, double b, double tolerance = 1e-7) {
  return std::abs(a - b) < tolerance;
}

template <typename T>
void flatten_vector4d(
    const std::vector<std::vector<std::vector<std::vector<T>>>> &vec4d,
    std::vector<T> &vec) {
  std::size_t idx = 0;
  for (const auto &cube : vec4d) {
    for (const auto &plane : cube) {
      for (const auto &row : plane) {
        for (const auto &val : row) {
          vec[idx++] = val;
        }
      }
    }
  }
}

template <typename T, std::size_t N>
void to_array(const std::vector<T> &vec, std::array<T, N> &arr) {
  std::copy(vec.begin(), vec.end(), arr.begin());
}

// Template function for 4D vector to 1D array
template <std::size_t N, typename T>
void vector4d_to_array(
    const std::vector<std::vector<std::vector<std::vector<T>>>> &vec4d,
    std::array<T, N> &arr) {
  std::size_t idx = 0;
  for (const auto &cube : vec4d) {
    for (const auto &plane : cube) {
      for (const auto &row : plane) {
        for (const auto &val : row) {
          if (idx < N) {
            arr[idx++] = val;
          }
        }
      }
    }
  }
  assert(idx <= N && "Vector has more elements than array can hold");
}

// std::pair<std::vector<double>, std::vector<double>>
// createFlatGrid(Rank1View<double, HostMemorySpace> xvec, Rank1View<double, HostMemorySpace>yvec) {
//     size_t nx = xvec.size();
//     size_t ny = yvec.size();
    
//     std::vector<double> X_flat, Y_flat;
//     X_flat.reserve(nx * ny);
//     Y_flat.reserve(nx * ny);
    
//     // Fill in row-major order: (x0,y0), (x1,y0), ..., (xn,y0), (x0,y1), ...
//     for (size_t i = 0; i < ny; ++i) {
//         for (size_t j = 0; j < nx; ++j) {
//             X_flat.push_back(xvec[j]);
//             Y_flat.push_back(yvec[i]);
//         }
//     }
    
//     return {X_flat, Y_flat};
//   }

void createFlatGrid(Rank1View<double, HostMemorySpace> xvec, Rank1View<double, HostMemorySpace> yvec,
  Kokkos::View<double*, HostMemorySpace> X_flat, Kokkos::View<double*, HostMemorySpace> Y_flat) {
    size_t nx = xvec.size();
    size_t ny = yvec.size();
    
    // Fill in row-major order: (x0,y0), (x1,y0), ..., (xn,y0), (x0,y1), ...
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            size_t idx = i * nx + j;
            X_flat[idx] = xvec[j];
            Y_flat[idx] = yvec[i];
        }
    }
}

void tset(int nth, std::vector<double> &th, std::vector<double> &sth,
          std::vector<double> &cth, double thmin, double thmax) {
  for (int ith = 0; ith < nth; ++ith) {
    th[ith] = thmin + static_cast<double>(ith) * (thmax - thmin) /
                          static_cast<double>(nth - 1);
    sth[ith] = 2.0 + std::sin(th[ith]);
    cth[ith] = std::cos(th[ith]);
  }
}

void xset(int nx, std::vector<double> &x, std::vector<double> &ex, double xmin,
          double xmax) {
  if (nx < 2)
    return; // avoid division by zero

  for (int ix = 0; ix < nx; ++ix) {
    x[ix] = xmin + static_cast<double>(ix) * (xmax - xmin) /
                       static_cast<double>(nx - 1);
    ex[ix] = std::exp(2.0 * x[ix] - 1.0);
  }
}

// void ffset(int num, const std::vector<double>& xf, const std::vector<double>&
// tf,
//            std::vector<std::vector<std::vector<std::vector<double>>>>& f) {
//     for (int j = 0; j < num; ++j) {
//         for (int i = 0; i < num; ++i) {
//             f[0][0][i][j] = xf[i] * tf[j];
//         }
//     }
// }

void ffset(int num, const std::vector<double> &xf,
           const std::vector<double> &tf, std::vector<double> &f) {
  for (int j = 0; j < num; ++j) {
    for (int i = 0; i < num; ++i) {
      f[i * num + j] = xf[i] * tf[j];
    }
  }
}

void bset(Rank1View<double, HostMemorySpace> fx, int nx,
          Rank1View<double, HostMemorySpace> fth, int nth,
          Rank1View<double, HostMemorySpace> bcx1,
          Rank1View<double, HostMemorySpace> bcx2,
          Rank1View<double, HostMemorySpace> bcth1,
          Rank1View<double, HostMemorySpace> bcth2) {

  // df/dx = 2*exp(2x-1)*(2+sin(th)) = 2*f  (represented using fx*fth)
  for (int ith = 0; ith < nth; ++ith) {
    bcx1[ith] = 2.0 * fx[0] * fth[ith];      // df/dx at x(1)
    bcx2[ith] = 2.0 * fx[nx - 1] * fth[ith]; // df/dx at x(nx)
  }

  // df/dth = exp(2x-1)*cos(th) → cos(0) = cos(2π) = 1 → df/dth = fx[ix]
  for (int ix = 0; ix < nx; ++ix) {
    bcth1[ix] = fx[ix]; // df/dth at th = 0 (th[0])
    bcth2[ix] = fx[ix]; // df/dth at th = 2π (th[nth - 1])
  }
}

void dotest1(int ns, Rank1View<double, HostMemorySpace> x,
             Rank1View<double, HostMemorySpace> f,
             Rank1View<double, HostMemorySpace> fd,
             Rank2View<double, HostMemorySpace> fspl,
             Rank2View<double, HostMemorySpace> fspp,
             Rank2View<double, HostMemorySpace> fs2, int nt,
             Rank1View<double, HostMemorySpace> xt,
             Rank1View<double, HostMemorySpace> ft,
             Rank2View<double, HostMemorySpace> xpkg,
             Rank2View<double, HostMemorySpace> testa1,
             Rank2View<double, HostMemorySpace> testa2,
             Rank2View<double, HostMemorySpace> testa3,
             Rank1View<double, HostMemorySpace> wk,
             Rank1View<double, HostMemorySpace> wk2) {
  using std::data;
  using std::size;
  int ierg = 0;
  int iwarn = 0;

  for (int i = 0; i < ns; ++i) {
    fspl(0, i) = f[i];
    fspp(0, i) = f[i];
    fs2(0, i) = f[i];
  }

  std::array<double, 40> fspl4_arr;
  auto fspl4 = Rank2View<double, HostMemorySpace>(fspl4_arr.data(), 4, 10);

  // CubicSplineInterpolator<double, HostMemorySpace> interpolator;

  // interpolator.genxpkg(ns, x, xpkg, 1, 1, 1, 4.0e-7, 1);
  // if (ierg != 0) {
  //     std::cerr << "Error in genxpkg: " << ierg << std::endl;
  //     return;
  // }

  int ilinx = 1; // Dummy interpolation lookup table
  int ier = 0;

  // interpolator.cspline(x, ns, fspl, 1, 1, 1, 1, wk);
  CubicSplineInterpolator<double, HostMemorySpace> explicit_interpolator(x, ns, fspl, 1, 1, 1, 1, wk);

  std::array<double, 3> fget_arr = {0.0, 0.0, 0.0};
  auto fget = Rank1View<double, HostMemorySpace>(fget_arr.data(), 3);
  double sdif = 0.0;
  double pdif = 0.0;
  double s2dif = 0.0;
  double sdifr = 0.0;
  double pdifr = 0.0;
  double s2difr = 0.0;
  double difabs = 0.0;
  // std::vector<int> ict = {1, 0, 0}; // Request only function value
  Kokkos::View<int*, HostMemorySpace> ict("ict", 3);
  Kokkos::deep_copy(ict, 0);
  ict[0] = 1;

  std::array<double, 3000> splinv_arr;
  auto splinv = Rank2View<double, HostMemorySpace>(splinv_arr.data(), 1000, 3);
  explicit_interpolator.evaluate_explicit(ict, xt, splinv);

  for (int i = 0; i < 1000; ++i) {
    explicit_interpolator.cspeval(xt[i], ict, fget, ier);
    if (ier != 0) {
      ier = 0;
    } else {
      assert(fget(0) == splinv(i, 0));
      difabs = std::abs(fget(0) - ft[i]);
      sdif = std::max(sdif, difabs);
      sdifr = std::max(sdifr, difabs / ft[i]);
    }
  }
  
  std::cout << "1d spline max absolute difference " << sdif << std::endl;
  assert(are_equal(sdif, 6.7572E-04));
  std::cout << "1d spline relative difference " << sdifr << std::endl;
  assert(are_equal(sdifr, 6.6529E-04));

  // interpolator.cspline(x, ns, fspp, -1, 0, -1, 0, wk);
  CubicSplineInterpolator<double, HostMemorySpace> explicit_interpolator_periodic(x, ns, fspp, -1, 0, -1, 0, wk);
  explicit_interpolator_periodic.evaluate_explicit(ict, xt, splinv);

  for (int i = 0; i < 1000; ++i) {
    explicit_interpolator_periodic.cspeval(xt[i], ict, fget, ier);
    if (ier != 0) {
      ier = 0;
    } else {
      assert(fget(0) == splinv(i, 0));
      difabs = std::abs(fget(0) - ft[i]);
      pdif = std::max(pdif, difabs);
      pdifr = std::max(pdifr, difabs / ft[i]);
    }
  }

  std::cout << "1d periodic max absolute difference " << pdif << std::endl;
  assert(are_equal(pdif, 6.8669E-04));
  std::cout << "1d periodic relative difference " << pdifr << std::endl;
  assert(are_equal(pdifr, 6.7622E-04));

  // interpolator.mkspline(x, ns, fs2, fspl4, 1, 1, 1, 1, wk2);
  CubicSplineInterpolator<double, HostMemorySpace> compact_interpolator(x, ns, fs2, fspl4, 1, 1, 1, 1, wk2);
  compact_interpolator.evaluate_compact(ict, xt, splinv);
  for (int i = 0; i < 1000; ++i) {
    compact_interpolator.evspline(xt[i], ict, fget, ier);
    if (ier != 0) {
      ier = 0;
    } else {
      assert(fget(0) == splinv(i, 0));
      difabs = std::abs(fget(0) - ft[i]);
      s2dif = std::max(s2dif, difabs);
      s2difr = std::max(s2difr, difabs / ft[i]);
    }
  }

  std::cout << "1d spline2 max absolute difference " << s2dif << std::endl;
  assert(are_equal(s2dif, 6.7572E-04));
  std::cout << "1d spline2 relative difference " << s2difr << std::endl;
  assert(are_equal(s2difr, 6.6529E-04));
}

void pspltest1(double zctrl) {
  const double pi2 = 6.28318530718;
  const double zero = 0.0;
  int inum = 10;

  // Local arrays
  std::vector<double> zdum_vec(1000), testa1_vec(1000), testa2_vec(1000),
      testa3_vec(1000), xtest_vec(1000), ftest_vec(1000);
  std::vector<double> x_vec(10), zcos_vec(10), z2sin_vec(10);
  std::vector<double> xpkg_vec(10 * 4);
  std::vector<double> fs_vec(40), fsp_vec(40);
  std::vector<double> fs2_vec(20);

  // Prepare test data
  tset(1000, xtest_vec, ftest_vec, zdum_vec, zero - 0.1, pi2 + 0.1);
  tset(inum, x_vec, z2sin_vec, zcos_vec, zero, pi2);

  // Calculate derivative (df/dx = cos(x))
  for (int ix = 0; ix < inum; ++ix) {
    zcos_vec[ix] = std::cos(x_vec[ix]);
  }

  // std::array<double, 1000> zdum_arr;
  // to_array(zdum_vec, zdum_arr);
  auto zdum_arr = Kokkos::View<double*, HostMemorySpace>(
    zdum_vec.data(), zdum_vec.size());
  auto zdum =
      Rank1View<double, HostMemorySpace>(zdum_arr.data(), zdum_arr.size());
  // std::array<double, 1000> wk2_arr;
  // std::copy(zdum_vec.begin(), zdum_vec.end(), wk2_arr.begin());
  // Kokkos::View<double*, HostMemorySpace> wk2_arr("wk2_arr", 3);
  auto wk2_arr = Kokkos::View<double*, HostMemorySpace>(
    zdum_vec.data(), zdum_vec.size());
  auto wk2 = Rank1View<double, HostMemorySpace>(wk2_arr.data(), wk2_arr.size());
  // std::array<double, 3000> testa1_arr;
  // to_array(testa1_vec, testa1_arr);
  auto testa1_arr = Kokkos::View<double*, HostMemorySpace>(
    testa1_vec.data(), testa1_vec.size());
  auto testa1 = Rank2View<double, HostMemorySpace>(testa1_arr.data(), 1000, 3);
  // std::array<double, 3000> testa2_arr;
  // to_array(testa2_vec, testa2_arr);
  auto testa2_arr = Kokkos::View<double*, HostMemorySpace>(
    testa2_vec.data(), testa2_vec.size());
  auto testa2 = Rank2View<double, HostMemorySpace>(testa2_arr.data(), 1000, 3);
  // std::array<double, 3000> testa3_arr;
  // to_array(testa3_vec, testa3_arr);
  auto testa3_arr = Kokkos::View<double*, HostMemorySpace>(
    testa3_vec.data(), testa3_vec.size());
  auto testa3 = Rank2View<double, HostMemorySpace>(testa3_arr.data(), 1000, 3);
  // std::array<double, 10> x_arr;
  // to_array(x_vec, x_arr);
  auto x_arr = Kokkos::View<double*, HostMemorySpace>(
    x_vec.data(), x_vec.size());
  auto x = Rank1View<double, HostMemorySpace>(x_arr.data(), x_arr.size());
  // std::array<double, 10> zcos_arr;
  // to_array(zcos_vec, zcos_arr);
  auto zcos_arr = Kokkos::View<double*, HostMemorySpace>(
    zcos_vec.data(), zcos_vec.size());
  auto zcos =
      Rank1View<double, HostMemorySpace>(zcos_arr.data(), zcos_arr.size());
  // std::array<double, 10> z2sin_arr;
  // to_array(z2sin_vec, z2sin_arr);
  auto z2sin_arr = Kokkos::View<double*, HostMemorySpace>(
    z2sin_vec.data(), z2sin_vec.size());
  auto z2sin =
      Rank1View<double, HostMemorySpace>(z2sin_arr.data(), z2sin_arr.size());
  // std::array<double, 40> fs_arr;
  // to_array(fs_vec, fs_arr);
  auto fs_arr = Kokkos::View<double*, HostMemorySpace>(
    fs_vec.data(), fs_vec.size());
  auto fs = Rank2View<double, HostMemorySpace>(fs_arr.data(), 4, 10);
  // std::array<double, 40> fsp_arr;
  // to_array(fsp_vec, fsp_arr);
  auto fsp_arr = Kokkos::View<double*, HostMemorySpace>(
    fsp_vec.data(), fsp_vec.size());
  auto fsp = Rank2View<double, HostMemorySpace>(fsp_arr.data(), 4, 10);
  // std::array<double, 20> fs2_arr;
  // to_array(fs2_vec, fs2_arr);
  auto fs2_arr = Kokkos::View<double*, HostMemorySpace>(
    fs2_vec.data(), fs2_vec.size());
  auto fs2 = Rank2View<double, HostMemorySpace>(fs2_arr.data(), 2, 10);
  // std::array<double, 40> xpkg_arr;
  // to_array(xpkg_vec, xpkg_arr);
  auto xpkg_arr = Kokkos::View<double*, HostMemorySpace>(
    xpkg_vec.data(), xpkg_vec.size());
  auto xpkg = Rank2View<double, HostMemorySpace>(xpkg_arr.data(), 10, 4);
  // std::array<double, 1000> xtest_arr;
  // to_array(xtest_vec, xtest_arr);
  auto xtest_arr = Kokkos::View<double*, HostMemorySpace>(
    xtest_vec.data(), xtest_vec.size());
  auto xtest = Rank1View<double, HostMemorySpace>(xtest_arr.data(), 1000);
  // std::array<double, 1000> ftest_arr;
  // to_array(ftest_vec, ftest_arr);
  auto ftest_arr = Kokkos::View<double*, HostMemorySpace>(
    ftest_vec.data(), ftest_vec.size());
  auto ftest = Rank1View<double, HostMemorySpace>(ftest_arr.data(), 1000);

  // Call test function
  dotest1(inum, x, z2sin, zcos, fs, fsp, fs2, 1000, xtest, ftest, xpkg, testa1,
          testa2, testa3, zdum, wk2);
}

void compare(const std::string &slbl,
             Rank1View<double, HostMemorySpace> x, int nx,
             Rank1View<double, HostMemorySpace> th, int nth,
             Rank4View<double, HostMemorySpace> f,
             Rank3View<double, HostMemorySpace> fh,
             Rank2View<double, HostMemorySpace> fl, int ilinx, int ilinth,
             Rank1View<double, HostMemorySpace> xtest,
             Rank1View<double, HostMemorySpace> fxtest,
             Rank1View<double, HostMemorySpace> thtest,
             Rank1View<double, HostMemorySpace> fthtest, int ntest,
             BiCubicSplineInterpolator<double, HostMemorySpace> interpolator,
             Kokkos::View<int*, HostMemorySpace> isel, Rank2View<double, HostMemorySpace> splinv) {

  int icycle = 20;
  int iherm = 0;
  if (slbl == "hermite")
    iherm = 1;
  else if (slbl == "mkbicub")
    iherm = 2;
  else if (slbl == "piecewise linear")
    iherm = 3;

  double fmin = 1.0e30;
  double fmax = -1.0e30;
  double fdif = 0.0;
  double fdifr = 0.0;
  int ier;

  // std::vector<double> fget_vec(10);
  Kokkos::View<double*, HostMemorySpace> fget_vec("fget_vec", 10);
  auto fget = Rank1View<double, HostMemorySpace>(fget_vec.data(), 10);
  double zth = 0.0;
  double zx = 0.0;
  double ff = 0.0;

  auto splinv_reshaped = Rank3View<double, HostMemorySpace>(
      splinv.data_handle(), ntest, ntest, 10);


  for (int j = 0; j < ntest; ++j) {
    zth = thtest[j];
    for (int i = 0; i < ntest; ++i) {
      zx = xtest[i];
      ff = fxtest[i] * fthtest[j];
      fmin = std::min(fmin, ff);
      fmax = std::max(fmax, ff);

      if (iherm == 0) {
        interpolator.bcspeval(zx, zth, isel, fget, ier);
      } else if (iherm == 2) {
        interpolator.evbicub(zx, zth, isel, fget, ier);
      }

      if (ier == 0) {
        assert(fget(0) == splinv_reshaped(j, i, 0));
        double fs = fget(0); // Interpolated value
        fdif = std::max(fdif, std::abs(ff - fs));
        fdifr = std::max(fdifr, std::abs((ff - fs) / (0.5 * (ff + fs))));
      }
      
    }
  }
  std::cout << "2d" << slbl << "  min: " << fmin << "  max: " << fmax
            << "  dif: " << fdif << "  difr: " << fdifr << std::endl;
  assert(are_equal(fdif, 1.8312E-03));
  assert(are_equal(fdifr, 6.7151E-04));
}

void dotest2(Rank1View<double, HostMemorySpace> x,
             Rank1View<double, HostMemorySpace> fx, int nx,
             Rank1View<double, HostMemorySpace> th,
             Rank1View<double, HostMemorySpace> fth,
             Rank1View<double, HostMemorySpace> dfth, int nth,
             Rank4View<double, HostMemorySpace> f,
             Rank3View<double, HostMemorySpace> fh,
             Rank2View<double, HostMemorySpace> flin,
             Rank1View<double, HostMemorySpace> bcx1,
             Rank1View<double, HostMemorySpace> bcx2,
             Rank1View<double, HostMemorySpace> bcth1,
             Rank1View<double, HostMemorySpace> bcth2,
             Rank1View<double, HostMemorySpace> xtest,
             Rank1View<double, HostMemorySpace> fxtest,
             Rank1View<double, HostMemorySpace> thtest,
             Rank1View<double, HostMemorySpace> fthtest, int ntest) {
  for (int ith = 0; ith < nth; ++ith) {
    for (int ix = 0; ix < nx; ++ix) {
      flin(ix, ith) = f(0, 0, ix, ith);          // f
      fh(0, ix, ith) = f(0, 0, ix, ith);         // f
      fh(1, ix, ith) = 2.0 * f(0, 0, ix, ith);   // df/dx
      fh(2, ix, ith) = fx[ix] * dfth[ith];       // df/dy
      fh(3, ix, ith) = 2.0 * fx[ix] * dfth[ith]; // d2f/dxdy
    }
  }

  int ier = 0;
  int nbc = 1;
  int ilinx = 0;
  int ilinth = 0;
  // std::vector<double> wk_vec(1000);
  Kokkos::View<double*, HostMemorySpace> wk_vec("wk_vec", 1000);
  auto wk = Rank1View<double, HostMemorySpace>(wk_vec.data(), 1000);

  // BiCubicSplineInterpolator<double, HostMemorySpace> interpolator;

  bset(fx, nx, fth, nth, bcx1, bcx2, bcth1, bcth2);

  // // std::array<double, 40> fspl_x_arr;
  // Kokkos::View<double*, HostMemorySpace> fspl_x_arr(40);
  // auto fspl_x = Rank2View<double, HostMemorySpace>(fspl_x_arr.data(), 4, 10);
  // // std::array<double, 40> fspl_th_arr;
  // Kokkos::View<double*, HostMemorySpace> fspl_th_arr(40);
  // auto fspl_th = Rank2View<double, HostMemorySpace>(fspl_th_arr.data(), 4, 10);
  // std::array<double, 10> wk_x_arr;
  // auto wk_x = Rank1View<double, HostMemorySpace>(wk_x_arr.data(), 10);
  // std::array<double, 10> wk_th_arr;
  // auto wk_th = Rank1View<double, HostMemorySpace>(wk_th_arr.data(), 10);
  // std::array<double, 400> fspl_l_th_arr;
  // auto fspl_l_th =
  //     Rank2View<double, HostMemorySpace>(fspl_l_th_arr.data(), 40, 10);

  // interpolator.bcspline(x, nx, th, nth, f, nbc, bcx1, nbc, bcx2, nbc, bcth1,
  //                       nbc, bcth2, wk);
  BiCubicSplineInterpolator<double, HostMemorySpace> explicit_interpolator(
      x, nx, th, nth, f, nbc, bcx1, nbc, bcx2, nbc, bcth1, nbc, bcth2, wk);
  if (ier != 0) {
    std::cerr << " ?? error in pspltest:  dotest2(bcspline)\n";
    return;
  }
  // std::array<double, 400000> splinv_arr;
  Kokkos::View<double*, HostMemorySpace> splinv_arr("splinv_arr", 400000);
  auto splinv = Rank2View<double, HostMemorySpace>(splinv_arr.data(), 40000, 10);
  // std::vector<int> isel(10, 0);
  Kokkos::View<int*, HostMemorySpace> isel("isel", 10);
  Kokkos::deep_copy(isel, 0);
  isel[0] = 1;

  // auto [xtest_grid_vec, thtest_grid_vec] = createFlatGrid(xtest, thtest);
  // std::array<double, 40000> xtest_grid_arr;
  // std::array<double, 40000> thtest_grid_arr;
  // to_array(xtest_grid_vec, xtest_grid_arr);
  // to_array(thtest_grid_vec, thtest_grid_arr);
  // auto xtest_grid_arr = Kokkos::View<double*, HostMemorySpace>(xtest_grid_vec.data(), xtest_grid_vec.size());
  // auto thtest_grid_arr = Kokkos::View<double*, HostMemorySpace>(thtest_grid_vec.data(), thtest_grid_vec.size());
  Kokkos::View<double*, HostMemorySpace> xtest_grid_arr("xtest_grid_arr", 40000);
  Kokkos::View<double*, HostMemorySpace> thtest_grid_arr("thtest_grid_arr", 40000);
  createFlatGrid(xtest, thtest, xtest_grid_arr, thtest_grid_arr);
  auto xtest_grid =
      Rank1View<double, HostMemorySpace>(xtest_grid_arr.data(), 40000);
  auto thtest_grid =
      Rank1View<double, HostMemorySpace>(thtest_grid_arr.data(), 40000);

  explicit_interpolator.evaluate_explicit(isel, xtest_grid, thtest_grid, splinv);

  compare("bcspline", x, nx, th, nth, f, fh, flin, ilinx, ilinth, xtest, fxtest,
          thtest, fthtest, ntest, explicit_interpolator, isel, splinv);

  for (int ith = 0; ith < nth; ++ith) {
    for (int ix = 0; ix < nx; ++ix) {
      fh(0, ix, ith) = f(0, 0, ix, ith); // f only
    }
  }
  
  // interpolator.mkbicub(x, nx, th, nth, fh, nbc, bcx1, nbc, bcx2, nbc, bcth1,
  //                      nbc, bcth2, wk);
  BiCubicSplineInterpolator<double, HostMemorySpace> compact_interpolator(
      x, nx, th, nth, fh, nbc, bcx1, nbc, bcx2, nbc, bcth1, nbc, bcth2, wk);
  compact_interpolator.evaluate_compact(isel, xtest_grid, thtest_grid, splinv);

  compare("mkbicub", x, nx, th, nth, f, fh, flin, ilinx, ilinth, xtest, fxtest,
          thtest, fthtest, ntest, compact_interpolator, isel, splinv);
}

void pspltest2(double zctrl) {
  const double pi2 = 6.28318530718;
  const double zero = 0.0;
  const double one = 1.0;
  int inum = 10;

  std::vector<double> x1_vec(10), ex1_vec(10), t1_vec(10), st1_vec(10),
      ct1_vec(10);
  std::vector<double> x2_vec(20), ex2_vec(20), t2_vec(20), st2_vec(20),
      ct2_vec(20);
  std::vector<double> x4_vec(40), ex4_vec(40), t4_vec(40), st4_vec(40),
      ct4_vec(40);
  std::vector<double> xtest_vec(200), extest_vec(200), ttest_vec(200),
      stest_vec(200), ctest_vec(200);

  std::vector<double> f1_vec(1600), f2_vec(6400), f4_vec(4 * 4 * 40 * 40);
  std::vector<double> bcx1_vec(40), bcx2_vec(40), bcth1_vec(40), bcth2_vec(40);
  std::vector<double> fh_vec(400);
  std::vector<double> flin_vec(100, 100);

  xset(10, x1_vec, ex1_vec, zero, one);
  xset(20, x2_vec, ex2_vec, zero, one);
  xset(40, x4_vec, ex4_vec, zero, one);
  xset(200, xtest_vec, extest_vec, zero, one);

  tset(10, t1_vec, st1_vec, ct1_vec, zero, pi2);
  tset(20, t2_vec, st2_vec, ct2_vec, zero, pi2);
  tset(40, t4_vec, st4_vec, ct4_vec, zero, pi2);
  tset(200, ttest_vec, stest_vec, ctest_vec, zero, pi2);

  ffset(10, ex1_vec, st1_vec, f1_vec);
  ffset(20, ex2_vec, st2_vec, f2_vec);
  ffset(40, ex4_vec, st4_vec, f4_vec);

  // std::array<double, 10> x1_arr;
  // to_array(x1_vec, x1_arr);
  auto x1_arr = Kokkos::View<double*, HostMemorySpace>(x1_vec.data(), x1_vec.size());
  auto x1 =
      Rank1View<double, HostMemorySpace>(x1_arr.data(), x1_arr.size());

  // std::array<double, 10> ex1_arr;
  // to_array(ex1_vec, ex1_arr);
  auto ex1_arr = Kokkos::View<double*, HostMemorySpace>(ex1_vec.data(), ex1_vec.size());
  auto ex1 = Rank1View<double, HostMemorySpace>(ex1_arr.data(), ex1_arr.size());

  // td::array<double, 10> t1_arr;
  // to_array(t1_vec, t1_arr);s
  auto t1_arr = Kokkos::View<double*, HostMemorySpace>(t1_vec.data(), t1_vec.size());
  auto t1 =
      Rank1View<double, HostMemorySpace>(t1_arr.data(), t1_arr.size());

  // std::array<double, 10> st1_arr;
  // to_array(st1_vec, st1_arr);
  auto st1_arr = Kokkos::View<double*, HostMemorySpace>(st1_vec.data(), st1_vec.size());
  auto st1 = Rank1View<double, HostMemorySpace>(st1_arr.data(), st1_arr.size());
  // std::array<double, 10> ct1_arr;
  // to_array(ct1_vec, ct1_arr);
  auto ct1_arr = Kokkos::View<double*, HostMemorySpace>(ct1_vec.data(), ct1_vec.size());
  auto ct1 = Rank1View<double, HostMemorySpace>(ct1_arr.data(), ct1_arr.size());

  // std::array<double, 1600> f1_arr;
  // to_array(f1_vec, f1_arr);
  auto f1_arr = Kokkos::View<double*, HostMemorySpace>(f1_vec.data(), f1_vec.size());
  auto f1 = Rank4View<double, HostMemorySpace>(f1_arr.data(),
                                               Kokkos::extents{4, 4, 10, 10});
  // std::array<double, 1600> flin_arr;
  // to_array(flin_vec, flin_arr);
  auto flin_arr = Kokkos::View<double*, HostMemorySpace>(flin_vec.data(), flin_vec.size());
  auto flin = Rank2View<double, HostMemorySpace>(flin_arr.data(), 10, 10);
  // std::array<double, 6400> fh_arr;
  // to_array(fh_vec, fh_arr);
  auto fh_arr = Kokkos::View<double*, HostMemorySpace>(fh_vec.data(), fh_vec.size());
  auto fh = Rank3View<double, HostMemorySpace>(fh_arr.data(), 4, 10, 10);

  // std::array<double, 40> bcx1_arr;
  // to_array(bcx1_vec, bcx1_arr);
  auto bcx1_arr = Kokkos::View<double*, HostMemorySpace>(bcx1_vec.data(), bcx1_vec.size());
  auto bcx1 =
      Rank1View<double, HostMemorySpace>(bcx1_arr.data(), bcx1_arr.size());
  // std::array<double, 40> bcx2_arr;
  // to_array(bcx2_vec, bcx2_arr);
  auto bcx2_arr = Kokkos::View<double*, HostMemorySpace>(bcx2_vec.data(), bcx2_vec.size());
  auto bcx2 =
      Rank1View<double, HostMemorySpace>(bcx2_arr.data(), bcx2_arr.size());
  // std::array<double, 40> bcth1_arr;
  // to_array(bcth1_vec, bcth1_arr);
  auto bcth1_arr = Kokkos::View<double*, HostMemorySpace>(bcth1_vec.data(), bcth1_vec.size());
  auto bcth1 =
      Rank1View<double, HostMemorySpace>(bcth1_arr.data(), bcth1_arr.size());
  // std::array<double, 40> bcth2_arr;
  // to_array(bcth2_vec, bcth2_arr);
  auto bcth2_arr = Kokkos::View<double*, HostMemorySpace>(bcth2_vec.data(), bcth2_vec.size());
  auto bcth2 =
      Rank1View<double, HostMemorySpace>(bcth2_arr.data(), bcth2_arr.size());

  // std::array<double, 200> xtest_arr;
  // to_array(xtest_vec, xtest_arr);
  auto xtest_arr = Kokkos::View<double*, HostMemorySpace>(xtest_vec.data(), xtest_vec.size());
  auto xtest =
      Rank1View<double, HostMemorySpace>(xtest_arr.data(), xtest_arr.size());
  // std::array<double, 200> extest_arr;
  // to_array(extest_vec, extest_arr);
  auto extest_arr = Kokkos::View<double*, HostMemorySpace>(extest_vec.data(), extest_vec.size());
  auto extest =
      Rank1View<double, HostMemorySpace>(extest_arr.data(), extest_arr.size());
  // std::array<double, 200> ttest_arr;
  // to_array(ttest_vec, ttest_arr);
  auto ttest_arr = Kokkos::View<double*, HostMemorySpace>(ttest_vec.data(), ttest_vec.size());
  auto ttest =
      Rank1View<double, HostMemorySpace>(ttest_arr.data(), ttest_arr.size());
  // std::array<double, 200> stest_arr;
  // to_array(stest_vec, stest_arr);
  auto stest_arr = Kokkos::View<double*, HostMemorySpace>(stest_vec.data(), stest_vec.size());
  auto stest =
      Rank1View<double, HostMemorySpace>(stest_arr.data(), stest_arr.size());
  // std::array<double, 200> ctest_arr;
  // to_array(ctest_vec, ctest_arr);
  // auto ctest = Rank1View<double, HostMemorySpace>(ctest_arr.data(),
  // ctest_arr.size());

  dotest2(x1, ex1, 10, t1, st1, ct1, 10, f1, fh, flin, bcx1, bcx2, bcth1, bcth2,
          xtest, extest, ttest, stest, 200);

  // dotest2(x2,ex2,20,t2,st2,ct2,20,f2,fh,flin,
  //     bcx1,bcx2,bcth1,bcth2,
  //     xtest,extest,ttest,stest,200);

  // dotest2(x4,ex4,40,t4,st4,ct4,40,f4,fh,flin,
  //     bcx1,bcx2,bcth1,bcth2,
  //     xtest,extest,ttest,stest,200);
}

int main(int argc, char **argv) {
  // Example usage of the spline test
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);
  Kokkos::Timer timer;
  pspltest1(0.01);
  pspltest2(0.01);
  double time = timer.seconds();
  std::cout << "Total time for spline tests: " << time << " seconds"
            << std::endl;
  Kokkos::finalize();
  MPI_Finalize();
}