#include "mdspan/mdspan.hpp"
#include "pcms.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <pcms/interpolator/spline_interpolator.hpp>

using namespace pcms;

bool are_equal(double a, double b, double tolerance = 1e-7) {
  return std::abs(a - b) < tolerance;
}


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

void tset(int nth, Kokkos::View<double*, HostMemorySpace> th, Kokkos::View<double*, HostMemorySpace> sth,
          Kokkos::View<double*, HostMemorySpace> cth, double thmin, double thmax) {
  for (int ith = 0; ith < nth; ++ith) {
    th[ith] = thmin + static_cast<double>(ith) * (thmax - thmin) /
                          static_cast<double>(nth - 1);
    sth[ith] = 2.0 + std::sin(th[ith]);
    cth[ith] = std::cos(th[ith]);
  }
}

void xset(int nx, Kokkos::View<double*, HostMemorySpace> x, Kokkos::View<double*, HostMemorySpace> ex, double xmin,
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

void ffset(int num, Kokkos::View<double*, HostMemorySpace> xf,
           Kokkos::View<double*, HostMemorySpace> tf, Kokkos::View<double*, HostMemorySpace> f) {
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

  Kokkos::View<double*, HostMemorySpace> fspl4_view("fspl4_view", 40);
  auto fspl4 = Rank2View<double, HostMemorySpace>(fspl4_view.data(), 4, 10);

  // CubicSplineInterpolator<double, HostMemorySpace> interpolator;


  int ilinx = 1; // Dummy interpolation lookup table
  int ier = 0;

  // interpolator.cspline(x, ns, fspl, 1, 1, 1, 1, wk);
  CubicSplineInterpolator<double, HostMemorySpace> explicit_interpolator(x, ns, fspl, 1, 1, 1, 1, wk);

  Kokkos::View<double*, HostMemorySpace> fget_view("fget_view", 3);
  auto fget = Rank1View<double, HostMemorySpace>(fget_view.data(), 3);
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

  Kokkos::View<double*, HostMemorySpace> splinv_view("splinv_view", 3000);
  auto splinv = Rank2View<double, HostMemorySpace>(splinv_view.data(), 1000, 3);
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
  Kokkos::View<double*, HostMemorySpace> zdum_view("zdum_view", 1000);
  Kokkos::View<double*, HostMemorySpace> testa1_view("testa1_view", 1000);
  Kokkos::View<double*, HostMemorySpace> testa2_view("testa2_view", 1000);
  Kokkos::View<double*, HostMemorySpace> testa3_view("testa3_view", 1000);
  Kokkos::View<double*, HostMemorySpace> xtest_view("xtest_view", 1000);
  Kokkos::View<double*, HostMemorySpace> ftest_view("ftest_view", 1000);
  Kokkos::View<double*, HostMemorySpace> x_view("x_view", 10);
  Kokkos::View<double*, HostMemorySpace> zcos_view("zcos_view", 10);
  Kokkos::View<double*, HostMemorySpace> z2sin_view("z2sin_view", 10);
  Kokkos::View<double*, HostMemorySpace> xpkg_view("xpkg_view", 10 * 4);
  Kokkos::View<double*, HostMemorySpace> fs_view("fs_view", 40);
  Kokkos::View<double*, HostMemorySpace> fsp_view("fsp_view", 40);
  Kokkos::View<double*, HostMemorySpace> fs2_view("fs2_view", 20);


  // Prepare test data
  tset(1000, xtest_view, ftest_view, zdum_view, zero - 0.1, pi2 + 0.1);
  tset(inum, x_view, z2sin_view, zcos_view, zero, pi2);

  // Calculate derivative (df/dx = cos(x))
  for (int ix = 0; ix < inum; ++ix) {
    zcos_view[ix] = std::cos(x_view[ix]);
  }

 
  auto zdum =
      Rank1View<double, HostMemorySpace>(zdum_view.data(), zdum_view.size());

  auto wk2_view = Kokkos::View<double*, HostMemorySpace>(
    zdum_view.data(), 1000);
  auto wk2 = Rank1View<double, HostMemorySpace>(wk2_view.data(), wk2_view.size());

  auto testa1 = Rank2View<double, HostMemorySpace>(testa1_view.data(), 1000, 3);
  auto testa2 = Rank2View<double, HostMemorySpace>(testa2_view.data(), 1000, 3);
  auto testa3 = Rank2View<double, HostMemorySpace>(testa3_view.data(), 1000, 3);
  auto x = Rank1View<double, HostMemorySpace>(x_view.data(), x_view.size());
  auto zcos =
      Rank1View<double, HostMemorySpace>(zcos_view.data(), zcos_view.size());
  auto z2sin =
      Rank1View<double, HostMemorySpace>(z2sin_view.data(), z2sin_view.size());
  auto fs = Rank2View<double, HostMemorySpace>(fs_view.data(), 4, 10);
  auto fsp = Rank2View<double, HostMemorySpace>(fsp_view.data(), 4, 10);
  auto fs2 = Rank2View<double, HostMemorySpace>(fs2_view.data(), 2, 10);
  auto xpkg = Rank2View<double, HostMemorySpace>(xpkg_view.data(), 10, 4);
  auto xtest = Rank1View<double, HostMemorySpace>(xtest_view.data(), 1000);
  auto ftest = Rank1View<double, HostMemorySpace>(ftest_view.data(), 1000);

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


  // interpolator.bcspline(x, nx, th, nth, f, nbc, bcx1, nbc, bcx2, nbc, bcth1,
  //                       nbc, bcth2, wk);
  BiCubicSplineInterpolator<double, HostMemorySpace> explicit_interpolator(
      x, nx, th, nth, f, nbc, bcx1, nbc, bcx2, nbc, bcth1, nbc, bcth2, wk);
  if (ier != 0) {
    std::cerr << " ?? error in pspltest:  dotest2(bcspline)\n";
    return;
  }
  Kokkos::View<double*, HostMemorySpace> splinv_view("splinv_view", 400000);
  auto splinv = Rank2View<double, HostMemorySpace>(splinv_view.data(), 40000, 10);
  // std::vector<int> isel(10, 0);
  Kokkos::View<int*, HostMemorySpace> isel("isel", 10);
  Kokkos::deep_copy(isel, 0);
  isel[0] = 1;

  Kokkos::View<double*, HostMemorySpace> xtest_grid_view("xtest_grid_view", 40000);
  Kokkos::View<double*, HostMemorySpace> thtest_grid_view("thtest_grid_view", 40000);
  createFlatGrid(xtest, thtest, xtest_grid_view, thtest_grid_view);
  auto xtest_grid =
      Rank1View<double, HostMemorySpace>(xtest_grid_view.data(), 40000);
  auto thtest_grid =
      Rank1View<double, HostMemorySpace>(thtest_grid_view.data(), 40000);

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

  Kokkos::View<double*, HostMemorySpace> x1_view("x1_view", 10);
  Kokkos::View<double*, HostMemorySpace> ex1_view("ex1_view", 10);
  Kokkos::View<double*, HostMemorySpace> t1_view("t1_view", 10);
  Kokkos::View<double*, HostMemorySpace> st1_view("st1_view", 10);
  Kokkos::View<double*, HostMemorySpace> ct1_view("ct1_view", 10);
  Kokkos::View<double*, HostMemorySpace> xtest_view("xtest_view", 200);
  Kokkos::View<double*, HostMemorySpace> extest_view("extest_view", 200);
  Kokkos::View<double*, HostMemorySpace> ttest_view("ttest_view", 200);
  Kokkos::View<double*, HostMemorySpace> stest_view("stest_view", 200);
  Kokkos::View<double*, HostMemorySpace> ctest_view("ctest_view", 200);
  Kokkos::View<double*, HostMemorySpace> f1_view("f1_view", 1600);
  Kokkos::View<double*, HostMemorySpace> f2_view("f2_view", 6400);
  Kokkos::View<double*, HostMemorySpace> f4_view("f4_view", 4 * 4 * 40 * 40);
  Kokkos::View<double*, HostMemorySpace> bcx1_view("bcx1_view", 40);
  Kokkos::View<double*, HostMemorySpace> bcx2_view("bcx2_view", 40);
  Kokkos::View<double*, HostMemorySpace> bcth1_view("bcth1_view", 40);
  Kokkos::View<double*, HostMemorySpace> bcth2_view("bcth2_view", 40);
  Kokkos::View<double*, HostMemorySpace> fh_view("fh_view", 400);
  Kokkos::View<double*, HostMemorySpace> flin_view("flin_view", 100);


  xset(10, x1_view, ex1_view, zero, one);
  // xset(20, x2_view, ex2_view, zero, one);
  // xset(40, x4_view, ex4_view, zero, one);
  xset(200, xtest_view, extest_view, zero, one);

  tset(10, t1_view, st1_view, ct1_view, zero, pi2);
  // tset(20, t2_view, st2_view, ct2_view, zero, pi2);
  // tset(40, t4_view, st4_view, ct4_view, zero, pi2);
  tset(200, ttest_view, stest_view, ctest_view, zero, pi2);

  ffset(10, ex1_view, st1_view, f1_view);
  // ffset(20, ex2_view, st2_view, f2_view);
  // ffset(40, ex4_view, st4_view, f4_view);

  auto x1 =
      Rank1View<double, HostMemorySpace>(x1_view.data(), x1_view.size());
  auto ex1 = Rank1View<double, HostMemorySpace>(ex1_view.data(), ex1_view.size());
  auto t1 =
      Rank1View<double, HostMemorySpace>(t1_view.data(), t1_view.size());
  auto st1 = Rank1View<double, HostMemorySpace>(st1_view.data(), st1_view.size());
  auto ct1 = Rank1View<double, HostMemorySpace>(ct1_view.data(), ct1_view.size());
  auto f1 = Rank4View<double, HostMemorySpace>(f1_view.data(),
                                               Kokkos::extents{4, 4, 10, 10});
  auto flin = Rank2View<double, HostMemorySpace>(flin_view.data(), 10, 10);

  auto fh = Rank3View<double, HostMemorySpace>(fh_view.data(), 4, 10, 10);

  auto bcx1 =
      Rank1View<double, HostMemorySpace>(bcx1_view.data(), bcx1_view.size());
  auto bcx2 =
      Rank1View<double, HostMemorySpace>(bcx2_view.data(), bcx2_view.size());
  auto bcth1 =
      Rank1View<double, HostMemorySpace>(bcth1_view.data(), bcth1_view.size());
  auto bcth2 =
      Rank1View<double, HostMemorySpace>(bcth2_view.data(), bcth2_view.size());

  auto xtest =
      Rank1View<double, HostMemorySpace>(xtest_view.data(), xtest_view.size());
  auto extest =
      Rank1View<double, HostMemorySpace>(extest_view.data(), extest_view.size());
  auto ttest =
      Rank1View<double, HostMemorySpace>(ttest_view.data(), ttest_view.size());
  auto stest =
      Rank1View<double, HostMemorySpace>(stest_view.data(), stest_view.size());

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