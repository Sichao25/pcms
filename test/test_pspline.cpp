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

void dotest1(int ns, Rank1View<const double, HostMemorySpace> x,
             Rank1View<double, HostMemorySpace> f,
             Rank1View<double, HostMemorySpace> fd,
             Rank2View<double, HostMemorySpace> fspl,
             Rank2View<double, HostMemorySpace> fspp,
             Rank2View<double, HostMemorySpace> fs2, int nt,
             Rank1View<const double, HostMemorySpace> xt,
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

  CubicSplineInterpolator<double, HostMemorySpace> interpolator;

  // interpolator.genxpkg(ns, x, xpkg, 1, 1, 1, 4.0e-7, 1);
  // if (ierg != 0) {
  //     std::cerr << "Error in genxpkg: " << ierg << std::endl;
  //     return;
  // }

  int ilinx = 1; // Dummy interpolation lookup table
  int ier = 0;

  interpolator.cspline(x, ns, fspl, 1, 1, 1, 1, wk);

  // interpolator.spgrid(xt,1000,testa1,ns,xpkg,fspl,iwarn);
  // interpolator.spgrid(xt,1000,testa2,ns,xpkg,fspp,iwarn);
  // interpolator.gridspline(xt,1000,testa3,ns,xpkg,fs2,iwarn);

  std::array<double, 3> splinv_arr = {0.0, 0.0, 0.0};
  auto splinv = Rank1View<double, HostMemorySpace>(splinv_arr.data(), 3);
  double sdif = 0.0;
  double pdif = 0.0;
  double s2dif = 0.0;
  double sdifr = 0.0;
  double pdifr = 0.0;
  double s2difr = 0.0;
  double difabs = 0.0;
  std::vector<int> ict = {1, 0, 0}; // Request only function value
  for (int i = 0; i < 1000; ++i) {
    interpolator.cspeval(xt[i], ict, splinv, x, ns, ier);
    if (ier != 0) {
      ier = 0;
    } else {
      difabs = std::abs(splinv(0) - ft[i]);
      sdif = std::max(sdif, difabs);
      sdifr = std::max(sdifr, difabs / ft[i]);
    }
  }
  std::cout << "1d spline max absolute difference " << sdif << std::endl;
  assert(are_equal(sdif, 6.7572E-04));
  std::cout << "1d spline relative difference " << sdifr << std::endl;
  assert(are_equal(sdifr, 6.6529E-04));

  interpolator.cspline(x, ns, fspp, -1, 0, -1, 0, wk);

  for (int i = 0; i < 1000; ++i) {
    interpolator.cspeval(xt[i], ict, splinv, x, ns, ier);
    if (ier != 0) {
      ier = 0;
    } else {
      difabs = std::abs(splinv(0) - ft[i]);
      pdif = std::max(pdif, difabs);
      pdifr = std::max(pdifr, difabs / ft[i]);
    }
  }

  std::cout << "1d periodic max absolute difference " << pdif << std::endl;
  assert(are_equal(pdif, 6.8669E-04));
  std::cout << "1d periodic relative difference " << pdifr << std::endl;
  assert(are_equal(pdifr, 6.7622E-04));

  interpolator.mkspline(x, ns, fs2, fspl4, 1, 1, 1, 1, wk2);
  for (int i = 0; i < 1000; ++i) {
    interpolator.evspline(xt[i], ict, splinv, x, ns, ier);
    if (ier != 0) {
      ier = 0;
    } else {
      difabs = std::abs(splinv(0) - ft[i]);
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

  std::array<double, 1000> zdum_arr;
  to_array(zdum_vec, zdum_arr);
  auto zdum =
      Rank1View<double, HostMemorySpace>(zdum_arr.data(), zdum_arr.size());
  std::array<double, 1000> wk2_arr;
  std::copy(zdum_vec.begin(), zdum_vec.end(), wk2_arr.begin());
  auto wk2 = Rank1View<double, HostMemorySpace>(wk2_arr.data(), wk2_arr.size());
  std::array<double, 3000> testa1_arr;
  to_array(testa1_vec, testa1_arr);
  auto testa1 = Rank2View<double, HostMemorySpace>(testa1_arr.data(), 1000, 3);
  std::array<double, 3000> testa2_arr;
  to_array(testa2_vec, testa2_arr);
  auto testa2 = Rank2View<double, HostMemorySpace>(testa2_arr.data(), 1000, 3);
  std::array<double, 3000> testa3_arr;
  to_array(testa3_vec, testa3_arr);
  auto testa3 = Rank2View<double, HostMemorySpace>(testa3_arr.data(), 1000, 3);
  std::array<double, 10> x_arr;
  to_array(x_vec, x_arr);
  auto x = Rank1View<const double, HostMemorySpace>(x_arr.data(), x_arr.size());
  std::array<double, 10> zcos_arr;
  to_array(zcos_vec, zcos_arr);
  auto zcos =
      Rank1View<double, HostMemorySpace>(zcos_arr.data(), zcos_arr.size());
  std::array<double, 10> z2sin_arr;
  to_array(z2sin_vec, z2sin_arr);
  auto z2sin =
      Rank1View<double, HostMemorySpace>(z2sin_arr.data(), z2sin_arr.size());
  std::array<double, 40> fs_arr;
  to_array(fs_vec, fs_arr);
  auto fs = Rank2View<double, HostMemorySpace>(fs_arr.data(), 4, 10);
  std::array<double, 40> fsp_arr;
  to_array(fsp_vec, fsp_arr);
  auto fsp = Rank2View<double, HostMemorySpace>(fsp_arr.data(), 4, 10);
  std::array<double, 20> fs2_arr;
  to_array(fs2_vec, fs2_arr);
  auto fs2 = Rank2View<double, HostMemorySpace>(fs2_arr.data(), 2, 10);
  std::array<double, 40> xpkg_arr;
  to_array(xpkg_vec, xpkg_arr);
  auto xpkg = Rank2View<double, HostMemorySpace>(xpkg_arr.data(), 10, 4);
  std::array<double, 1000> xtest_arr;
  to_array(xtest_vec, xtest_arr);
  auto xtest = Rank1View<const double, HostMemorySpace>(xtest_arr.data(), 1000);
  std::array<double, 1000> ftest_arr;
  to_array(ftest_vec, ftest_arr);
  auto ftest = Rank1View<double, HostMemorySpace>(ftest_arr.data(), 1000);

  // Call test function
  dotest1(inum, x, z2sin, zcos, fs, fsp, fs2, 1000, xtest, ftest, xpkg, testa1,
          testa2, testa3, zdum, wk2);
}

void compare(const std::string &slbl,
             Rank1View<const double, HostMemorySpace> x, int nx,
             Rank1View<const double, HostMemorySpace> th, int nth,
             Rank4View<double, HostMemorySpace> f,
             Rank3View<double, HostMemorySpace> fh,
             Rank2View<double, HostMemorySpace> fl, int ilinx, int ilinth,
             Rank1View<double, HostMemorySpace> xtest,
             Rank1View<double, HostMemorySpace> fxtest,
             Rank1View<double, HostMemorySpace> thtest,
             Rank1View<double, HostMemorySpace> fthtest, int ntest,
             BiCubicSplineInterpolator<double, HostMemorySpace> interpolator) {

  std::vector<int> isel(10, 0);
  isel[0] = 1;

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

  std::vector<double> fget_vec(10);
  auto fget = Rank1View<double, HostMemorySpace>(fget_vec.data(), 10);
  double zth = 0.0;
  double zx = 0.0;
  double ff = 0.0;
  for (int j = 0; j < ntest; ++j) {
    zth = thtest[j];
    for (int i = 0; i < ntest; ++i) {
      zx = xtest[i];
      ff = fxtest[i] * fthtest[j];
      fmin = std::min(fmin, ff);
      fmax = std::max(fmax, ff);

      if (iherm == 0) {
        interpolator.bcspeval(zx, zth, isel, fget, x, nx, th, nth, ier);
      } else if (iherm == 2) {
        interpolator.evbicub(zx, zth, x, nx, th, nth, isel, fget, ier);
      }

      double fs = fget(0); // Interpolated value
      fdif = std::max(fdif, std::abs(ff - fs));
      fdifr = std::max(fdifr, std::abs((ff - fs) / (0.5 * (ff + fs))));
    }
  }
  std::cout << "2d" << slbl << "  min: " << fmin << "  max: " << fmax
            << "  dif: " << fdif << "  difr: " << fdifr << std::endl;
  assert(are_equal(fdif, 1.8312E-03));
  assert(are_equal(fdifr, 6.7151E-04));
}

void dotest2(Rank1View<const double, HostMemorySpace> x,
             Rank1View<double, HostMemorySpace> fx, int nx,
             Rank1View<const double, HostMemorySpace> th,
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
  std::vector<double> wk_vec(1000);
  auto wk = Rank1View<double, HostMemorySpace>(wk_vec.data(), 1000);

  BiCubicSplineInterpolator<double, HostMemorySpace> interpolator;

  bset(fx, nx, fth, nth, bcx1, bcx2, bcth1, bcth2);

  std::array<double, 40> fspl_x_arr;
  auto fspl_x = Rank2View<double, HostMemorySpace>(fspl_x_arr.data(), 4, 10);
  std::array<double, 40> fspl_th_arr;
  auto fspl_th = Rank2View<double, HostMemorySpace>(fspl_th_arr.data(), 4, 10);
  std::array<double, 10> wk_x_arr;
  auto wk_x = Rank1View<double, HostMemorySpace>(wk_x_arr.data(), 10);
  std::array<double, 10> wk_th_arr;
  auto wk_th = Rank1View<double, HostMemorySpace>(wk_th_arr.data(), 10);
  std::array<double, 400> fspl_l_th_arr;
  auto fspl_l_th =
      Rank2View<double, HostMemorySpace>(fspl_l_th_arr.data(), 40, 10);

  interpolator.bcspline(x, nx, th, nth, f, nbc, bcx1, nbc, bcx2, nbc, bcth1,
                        nbc, bcth2, wk);
  if (ier != 0) {
    std::cerr << " ?? error in pspltest:  dotest2(bcspline)\n";
    return;
  }

  compare("bcspline", x, nx, th, nth, f, fh, flin, ilinx, ilinth, xtest, fxtest,
          thtest, fthtest, ntest, interpolator);

  for (int ith = 0; ith < nth; ++ith) {
    for (int ix = 0; ix < nx; ++ix) {
      fh(0, ix, ith) = f(0, 0, ix, ith); // f only
    }
  }

  interpolator.mkbicub(x, nx, th, nth, fh, nbc, bcx1, nbc, bcx2, nbc, bcth1,
                       nbc, bcth2, wk);

  compare("mkbicub", x, nx, th, nth, f, fh, flin, ilinx, ilinth, xtest, fxtest,
          thtest, fthtest, ntest, interpolator);
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
  // std::vector<std::vector<std::vector<std::vector<double>>>> f1_vec(4,
  // std::vector<std::vector<std::vector<double>>>(4,
  // std::vector<std::vector<double>>(10, std::vector<double>(10))));
  // std::vector<std::vector<std::vector<std::vector<double>>>> f2_vec(4,
  // std::vector<std::vector<std::vector<double>>>(4,
  // std::vector<std::vector<double>>(20, std::vector<double>(20))));
  // std::vector<std::vector<std::vector<std::vector<double>>>> f4_vec(4,
  // std::vector<std::vector<std::vector<double>>>(4,
  // std::vector<std::vector<double>>(40, std::vector<double>(40))));
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

  std::array<double, 10> x1_arr;
  to_array(x1_vec, x1_arr);
  auto x1 =
      Rank1View<const double, HostMemorySpace>(x1_arr.data(), x1_arr.size());

  std::array<double, 10> ex1_arr;
  to_array(ex1_vec, ex1_arr);
  auto ex1 = Rank1View<double, HostMemorySpace>(ex1_arr.data(), ex1_arr.size());

  std::array<double, 10> t1_arr;
  to_array(t1_vec, t1_arr);
  auto t1 =
      Rank1View<const double, HostMemorySpace>(t1_arr.data(), t1_arr.size());

  std::array<double, 10> st1_arr;
  to_array(st1_vec, st1_arr);
  auto st1 = Rank1View<double, HostMemorySpace>(st1_arr.data(), st1_arr.size());
  std::array<double, 10> ct1_arr;
  to_array(ct1_vec, ct1_arr);
  auto ct1 = Rank1View<double, HostMemorySpace>(ct1_arr.data(), ct1_arr.size());

  std::array<double, 1600> f1_arr;
  to_array(f1_vec, f1_arr);
  auto f1 = Rank4View<double, HostMemorySpace>(f1_arr.data(),
                                               Kokkos::extents{4, 4, 10, 10});
  std::array<double, 1600> flin_arr;
  to_array(flin_vec, flin_arr);
  auto flin = Rank2View<double, HostMemorySpace>(flin_arr.data(), 10, 10);
  std::array<double, 6400> fh_arr;
  to_array(fh_vec, fh_arr);
  auto fh = Rank3View<double, HostMemorySpace>(fh_arr.data(), 4, 10, 10);

  std::array<double, 40> bcx1_arr;
  to_array(bcx1_vec, bcx1_arr);
  auto bcx1 =
      Rank1View<double, HostMemorySpace>(bcx1_arr.data(), bcx1_arr.size());
  std::array<double, 40> bcx2_arr;
  to_array(bcx2_vec, bcx2_arr);
  auto bcx2 =
      Rank1View<double, HostMemorySpace>(bcx2_arr.data(), bcx2_arr.size());
  std::array<double, 40> bcth1_arr;
  to_array(bcth1_vec, bcth1_arr);
  auto bcth1 =
      Rank1View<double, HostMemorySpace>(bcth1_arr.data(), bcth1_arr.size());
  std::array<double, 40> bcth2_arr;
  to_array(bcth2_vec, bcth2_arr);
  auto bcth2 =
      Rank1View<double, HostMemorySpace>(bcth2_arr.data(), bcth2_arr.size());

  std::array<double, 200> xtest_arr;
  to_array(xtest_vec, xtest_arr);
  auto xtest =
      Rank1View<double, HostMemorySpace>(xtest_arr.data(), xtest_arr.size());
  std::array<double, 200> extest_arr;
  to_array(extest_vec, extest_arr);
  auto extest =
      Rank1View<double, HostMemorySpace>(extest_arr.data(), extest_arr.size());
  std::array<double, 200> ttest_arr;
  to_array(ttest_vec, ttest_arr);
  auto ttest =
      Rank1View<double, HostMemorySpace>(ttest_arr.data(), ttest_arr.size());
  std::array<double, 200> stest_arr;
  to_array(stest_vec, stest_arr);
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