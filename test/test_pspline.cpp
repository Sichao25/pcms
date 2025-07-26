#include "mdspan/mdspan.hpp"
#include "pcms.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <pcms/interpolator/spline_interpolator.hpp>

using namespace pcms;

using TestMemorySpace = Kokkos::DefaultExecutionSpace;

bool are_equal(double a, double b, double tolerance = 1e-7) {
  return std::abs(a - b) < tolerance;
}

void print_mdspan(Rank1View<double, TestMemorySpace> view) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, view.size()),
      KOKKOS_LAMBDA(const int i) { printf("view[%d] = %f\n", i, view(i)); });
}

void print_2dmdspan(Rank2View<double, TestMemorySpace> view) {
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                             {view.extent(0), view.extent(1)}),
      KOKKOS_LAMBDA(const int i, const int j) {
        printf("view[%d][%d] = %f\n", i, j, view(i, j));
      });
}

void print_4dmdspan(Rank4View<double, TestMemorySpace> view) {
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, 0, 0, 0},
          {view.extent(0), view.extent(1), view.extent(2), view.extent(3)}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        printf("view[%d][%d][%d][%d] = %f\n", i, j, k, l, view(i, j, k, l));
      });
}

void print_view(Kokkos::View<double *, TestMemorySpace> view) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, view.size()),
      KOKKOS_LAMBDA(const int i) { printf("view[%d] = %f\n", i, view(i)); });
}

void createFlatGrid(Rank1View<double, TestMemorySpace> xvec,
                    Rank1View<double, TestMemorySpace> yvec,
                    Kokkos::View<double *, TestMemorySpace> X_flat,
                    Kokkos::View<double *, TestMemorySpace> Y_flat) {
  size_t nx = xvec.size();
  size_t ny = yvec.size();
  // for (size_t i = 0; i < ny; ++i) {
  //     for (size_t j = 0; j < nx; ++j) {
  //         size_t idx = i * nx + j;
  //         X_flat[idx] = xvec[j];
  //         Y_flat[idx] = yvec[i];
  //     }
  // }
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
      KOKKOS_LAMBDA(const int i, const int j) {
        size_t idx = i * nx + j;
        X_flat[idx] = xvec[j];
        Y_flat[idx] = yvec[i];
      });
}

void tset(int nth, Kokkos::View<double *, TestMemorySpace> th,
          Kokkos::View<double *, TestMemorySpace> sth,
          Kokkos::View<double *, TestMemorySpace> cth, double thmin,
          double thmax) {
  // for (int ith = 0; ith < nth; ++ith) {
  //   th[ith] = thmin + static_cast<double>(ith) * (thmax - thmin) /
  //                         static_cast<double>(nth - 1);
  //   sth[ith] = 2.0 + std::sin(th[ith]);
  //   cth[ith] = std::cos(th[ith]);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, nth),
      KOKKOS_LAMBDA(const int ith) {
        th[ith] = thmin + static_cast<double>(ith) * (thmax - thmin) /
                              static_cast<double>(nth - 1);
        sth[ith] = 2.0 + std::sin(th[ith]);
        cth[ith] = std::cos(th[ith]);
      });
}

void xset(int nx, Kokkos::View<double *, TestMemorySpace> x,
          Kokkos::View<double *, TestMemorySpace> ex, double xmin,
          double xmax) {
  if (nx < 2)
    return; // avoid division by zero

  // for (int ix = 0; ix < nx; ++ix) {
  //   x[ix] = xmin + static_cast<double>(ix) * (xmax - xmin) /
  //                      static_cast<double>(nx - 1);
  //   ex[ix] = std::exp(2.0 * x[ix] - 1.0);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, nx), KOKKOS_LAMBDA(const int ix) {
        x[ix] = xmin + static_cast<double>(ix) * (xmax - xmin) /
                           static_cast<double>(nx - 1);
        ex[ix] = std::exp(2.0 * x[ix] - 1.0);
      });
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

void ffset(int num, Kokkos::View<double *, TestMemorySpace> xf,
           Kokkos::View<double *, TestMemorySpace> tf,
           Kokkos::View<double *, TestMemorySpace> f) {
  // for (int j = 0; j < num; ++j) {
  //   for (int i = 0; i < num; ++i) {
  //     f[i * num + j] = xf[i] * tf[j];
  //   }
  // }
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num, num}),
      KOKKOS_LAMBDA(const int i, const int j) {
        f(i * num + j) = xf[i] * tf[j];
      });
}

void bset(Rank1View<double, TestMemorySpace> fx, int nx,
          Rank1View<double, TestMemorySpace> fth, int nth,
          Rank1View<double, TestMemorySpace> bcx1,
          Rank1View<double, TestMemorySpace> bcx2,
          Rank1View<double, TestMemorySpace> bcth1,
          Rank1View<double, TestMemorySpace> bcth2) {

  // df/dx = 2*exp(2x-1)*(2+sin(th)) = 2*f  (represented using fx*fth)
  // for (int ith = 0; ith < nth; ++ith) {
  //   bcx1[ith] = 2.0 * fx[0] * fth[ith];      // df/dx at x(1)
  //   bcx2[ith] = 2.0 * fx[nx - 1] * fth[ith]; // df/dx at x(nx)
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, nth),
      KOKKOS_LAMBDA(const int ith) {
        bcx1[ith] = 2.0 * fx(0) * fth(ith);      // df/dx at x(1)
        bcx2[ith] = 2.0 * fx(nx - 1) * fth(ith); // df/dx at x(nx)
      });

  // df/dth = exp(2x-1)*cos(th) → cos(0) = cos(2π) = 1 → df/dth = fx[ix]
  // for (int ix = 0; ix < nx; ++ix) {
  //   bcth1[ix] = fx[ix]; // df/dth at th = 0 (th[0])
  //   bcth2[ix] = fx[ix]; // df/dth at th = 2π (th[nth - 1])
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, nx), KOKKOS_LAMBDA(const int ix) {
        bcth1(ix) = fx(ix); // df/dth at th = 0 (th[0])
        bcth2(ix) = fx(ix); // df/dth at th = 2π (th[nth - 1])
      });
}

void dotest1(int ns, Rank1View<double, TestMemorySpace> x,
             Rank1View<double, TestMemorySpace> f,
             Rank1View<double, TestMemorySpace> fd,
             Rank2View<double, TestMemorySpace> fspl,
             Rank2View<double, TestMemorySpace> fspp,
             Rank2View<double, TestMemorySpace> fs2, int nt,
             Rank1View<double, TestMemorySpace> xt,
             Rank1View<double, TestMemorySpace> ft,
             Rank2View<double, TestMemorySpace> xpkg,
             Rank2View<double, TestMemorySpace> testa1,
             Rank2View<double, TestMemorySpace> testa2,
             Rank2View<double, TestMemorySpace> testa3,
             Rank1View<double, TestMemorySpace> wk,
             Rank1View<double, TestMemorySpace> wk2) {
  using std::data;
  using std::size;

  // for (int i = 0; i < ns; ++i) {
  //   fspl(0, i) = f[i];
  //   fspp(0, i) = f[i];
  //   fs2(0, i) = f[i];
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, ns), KOKKOS_LAMBDA(const int i) {
        fspl(0, i) = f(i);
        fspp(0, i) = f(i);
        fs2(0, i) = f(i);
      });

  Kokkos::View<double *, TestMemorySpace> fspl4_view("fspl4_view", 40);
  auto fspl4 = Rank2View<double, TestMemorySpace>(fspl4_view.data(), 4, 10);

  // interpolator.cspline(x, ns, fspl, 1, 1, 1, 1, wk);
  ExplicitCubicSplineInterpolator<double, TestMemorySpace>
      explicit_interpolator(x, ns, fspl, 1, 1, 1, 1, wk);

  double sdif = 0.0;
  double pdif = 0.0;
  double s2dif = 0.0;
  double sdifr = 0.0;
  double pdifr = 0.0;
  double s2difr = 0.0;
  int ict_arr[3] = {1, 0, 0};
  Kokkos::View<int *, TestMemorySpace> ict("ict", 3);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, ict.size()),
      KOKKOS_LAMBDA(const int i) { ict(i) = ict_arr[i]; });

  Kokkos::View<double *, TestMemorySpace> splinv_view("splinv_view", 3000);
  auto splinv = Rank2View<double, TestMemorySpace>(splinv_view.data(), 1000, 3);
  // explicit_interpolator.evaluate_explicit(ict, xt, splinv);

  //   double difabs = 0.0;
  //   for (int i = 0; i < 1000; ++i) {
  //     ExplicitCubicSplineInterpolator<double,
  //     TestMemorySpace>::cspeval(xt[i], ict, fget, x, ns, fspl, ier); if (ier
  //     != 0) {
  //       ier = 0;
  //     } else {
  //       difabs = std::abs(fget(0) - ft[i]);
  //       sdif = std::max(sdif, difabs);
  //       sdifr = std::max(sdifr, difabs / ft[i]);
  //     }
  //   }
  Kokkos::View<double *, TestMemorySpace> fget_view("fget_view", 3 * 1000);
  double result_sdif = 0.0;
  double result_sdifr = 0.0;
  Kokkos::parallel_reduce(
      "spline_eval", 1000,
      KOKKOS_LAMBDA(int i, double &max_sdif, double &max_sdifr) {
        int local_ier = 0;
        auto fget =
            Rank1View<double, TestMemorySpace>(fget_view.data() + i * 3, 3);
        ExplicitCubicSplineInterpolator<double, TestMemorySpace>::cspeval(
            xt(i), ict, fget, x, ns, fspl, local_ier);
        if (local_ier == 0) {
          double difabs = Kokkos::abs(fget(0) - ft(i));
          max_sdif = Kokkos::max(max_sdif, difabs);
          max_sdifr = Kokkos::max(max_sdifr, difabs / ft(i));
        }
      },
      Kokkos::Max<double>(result_sdif), Kokkos::Max<double>(result_sdifr));
  sdif = std::max(sdif, result_sdif);
  sdifr = std::max(sdifr, result_sdifr);

  std::cout << "1d spline max absolute difference " << sdif << std::endl;
  std::cout << "1d spline relative difference " << sdifr << std::endl;
  assert(are_equal(sdif, 6.7572E-04));
  assert(are_equal(sdifr, 6.6529E-04));

  // interpolator.cspline(x, ns, fspp, -1, 0, -1, 0, wk);
  ExplicitCubicSplineInterpolator<double, TestMemorySpace>
      explicit_interpolator_periodic(x, ns, fspp, -1, 0, -1, 0, wk);
  // explicit_interpolator_periodic.evaluate_explicit(ict, xt, splinv);

  // for (int i = 0; i < 1000; ++i) {
  //   ExplicitCubicSplineInterpolator<double, TestMemorySpace>::cspeval(xt[i],
  //   ict, fget, x, ns, fspp, ier); if (ier != 0) {
  //     ier = 0;
  //   } else {
  //     difabs = std::abs(fget(0) - ft[i]);
  //     pdif = std::max(pdif, difabs);
  //     pdifr = std::max(pdifr, difabs / ft[i]);
  //   }
  // }
  double result_pdif = 0.0;
  double result_pdifr = 0.0;

  Kokkos::parallel_reduce(
      "spline_eval", 1000,
      KOKKOS_LAMBDA(int i, double &max_pdif, double &max_pdifr) {
        int local_ier = 0;
        auto fget =
            Rank1View<double, TestMemorySpace>(fget_view.data() + i * 3, 3);
        ExplicitCubicSplineInterpolator<double, TestMemorySpace>::cspeval(
            xt[i], ict, fget, x, ns, fspp, local_ier);

        if (local_ier == 0) {
          double difabs =
              Kokkos::abs(fget(0) - ft[i]); // Use Kokkos::abs for device
          max_pdif = Kokkos::max(max_pdif, difabs);
          max_pdifr = Kokkos::max(max_pdifr, difabs / ft[i]);
        }
      },
      Kokkos::Max<double>(result_pdif), Kokkos::Max<double>(result_pdifr));

  pdif = std::max(pdif, result_pdif);
  pdifr = std::max(pdifr, result_pdifr);

  std::cout << "1d periodic max absolute difference " << pdif << std::endl;
  std::cout << "1d periodic relative difference " << pdifr << std::endl;
  assert(are_equal(pdif, 6.8669E-04));
  assert(are_equal(pdifr, 6.7622E-04));

  // interpolator.mkspline(x, ns, fs2, fspl4, 1, 1, 1, 1, wk2);
  CompactCubicSplineInterpolator<double, TestMemorySpace> compact_interpolator(
      x, ns, fs2, fspl4, 1, 1, 1, 1, wk2);
  // compact_interpolator.evaluate_compact(ict, xt, splinv);
  // for (int i = 0; i < 1000; ++i) {
  //   CompactubicSplineInterpolator<double, TestMemorySpace>::evspline(xt[i],
  //   ict, fget, x, ns, fs2, ier); if (ier != 0) {
  //     ier = 0;
  //   } else {
  //     difabs = std::abs(fget(0) - ft[i]);
  //     s2dif = std::max(s2dif, difabs);
  //     s2difr = std::max(s2difr, difabs / ft[i]);
  //   }
  // }
  double result_s2dif = 0.0;
  double result_s2difr = 0.0;
  Kokkos::parallel_reduce(
      "spline_eval_compact", 1000,
      KOKKOS_LAMBDA(int i, double &max_s2dif, double &max_s2difr) {
        int local_ier = 0;
        auto fget =
            Rank1View<double, TestMemorySpace>(fget_view.data() + i * 3, 3);
        CompactCubicSplineInterpolator<double, TestMemorySpace>::evspline(
            xt[i], ict, fget, x, ns, fs2, local_ier);

        if (local_ier == 0) {
          double difabs =
              Kokkos::abs(fget(0) - ft[i]); // Use Kokkos::abs for device
          max_s2dif = Kokkos::max(max_s2dif, difabs);
          max_s2difr = Kokkos::max(max_s2difr, difabs / ft[i]);
        }
      },
      Kokkos::Max<double>(result_s2dif), Kokkos::Max<double>(result_s2difr));
  s2dif = std::max(s2dif, result_s2dif);
  s2difr = std::max(s2difr, result_s2difr);

  std::cout << "1d spline2 max absolute difference " << s2dif << std::endl;
  std::cout << "1d spline2 relative difference " << s2difr << std::endl;
  assert(are_equal(s2dif, 6.7572E-04));
  assert(are_equal(s2difr, 6.6529E-04));
}

void pspltest1(double zctrl) {
  const double pi2 = 6.28318530718;
  const double zero = 0.0;
  int inum = 10;

  // Local arrays
  Kokkos::View<double *, TestMemorySpace> zdum_view("zdum_view", 1000);
  Kokkos::View<double *, TestMemorySpace> testa1_view("testa1_view", 1000);
  Kokkos::View<double *, TestMemorySpace> testa2_view("testa2_view", 1000);
  Kokkos::View<double *, TestMemorySpace> testa3_view("testa3_view", 1000);
  Kokkos::View<double *, TestMemorySpace> xtest_view("xtest_view", 1000);
  Kokkos::View<double *, TestMemorySpace> ftest_view("ftest_view", 1000);
  Kokkos::View<double *, TestMemorySpace> x_view("x_view", 10);
  Kokkos::View<double *, TestMemorySpace> zcos_view("zcos_view", 10);
  Kokkos::View<double *, TestMemorySpace> z2sin_view("z2sin_view", 10);
  Kokkos::View<double *, TestMemorySpace> xpkg_view("xpkg_view", 10 * 4);
  Kokkos::View<double *, TestMemorySpace> fs_view("fs_view", 40);
  Kokkos::View<double *, TestMemorySpace> fsp_view("fsp_view", 40);
  Kokkos::View<double *, TestMemorySpace> fs2_view("fs2_view", 20);

  // Prepare test data
  tset(1000, xtest_view, ftest_view, zdum_view, zero - 0.1, pi2 + 0.1);
  tset(inum, x_view, z2sin_view, zcos_view, zero, pi2);

  // Calculate derivative (df/dx = cos(x))
  // for (int ix = 0; ix < inum; ++ix) {
  //   zcos_view[ix] = std::cos(x_view[ix]);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, inum),
      KOKKOS_LAMBDA(const int ix) { zcos_view(ix) = std::cos(x_view(ix)); });

  auto zdum =
      Rank1View<double, TestMemorySpace>(zdum_view.data(), zdum_view.size());

  auto wk2_view =
      Kokkos::View<double *, TestMemorySpace>(zdum_view.data(), 1000);
  auto wk2 =
      Rank1View<double, TestMemorySpace>(wk2_view.data(), wk2_view.size());

  auto testa1 = Rank2View<double, TestMemorySpace>(testa1_view.data(), 1000, 3);
  auto testa2 = Rank2View<double, TestMemorySpace>(testa2_view.data(), 1000, 3);
  auto testa3 = Rank2View<double, TestMemorySpace>(testa3_view.data(), 1000, 3);
  auto x = Rank1View<double, TestMemorySpace>(x_view.data(), x_view.size());
  auto zcos =
      Rank1View<double, TestMemorySpace>(zcos_view.data(), zcos_view.size());
  auto z2sin =
      Rank1View<double, TestMemorySpace>(z2sin_view.data(), z2sin_view.size());
  auto fs = Rank2View<double, TestMemorySpace>(fs_view.data(), 4, 10);
  auto fsp = Rank2View<double, TestMemorySpace>(fsp_view.data(), 4, 10);
  auto fs2 = Rank2View<double, TestMemorySpace>(fs2_view.data(), 2, 10);
  auto xpkg = Rank2View<double, TestMemorySpace>(xpkg_view.data(), 10, 4);
  auto xtest = Rank1View<double, TestMemorySpace>(xtest_view.data(), 1000);
  auto ftest = Rank1View<double, TestMemorySpace>(ftest_view.data(), 1000);

  // Call test function
  print_view(x_view);
  dotest1(inum, x, z2sin, zcos, fs, fsp, fs2, 1000, xtest, ftest, xpkg, testa1,
          testa2, testa3, zdum, wk2);
}

void compare(const std::string &slbl, Rank1View<double, TestMemorySpace> x,
             int nx, Rank1View<double, TestMemorySpace> th, int nth,
             Rank4View<double, TestMemorySpace> f,
             Rank3View<double, TestMemorySpace> fh,
             Rank2View<double, TestMemorySpace> fl, int ilinx, int ilinth,
             Rank1View<double, TestMemorySpace> xtest,
             Rank1View<double, TestMemorySpace> fxtest,
             Rank1View<double, TestMemorySpace> thtest,
             Rank1View<double, TestMemorySpace> fthtest, int ntest,
             CubicSplineInterpolator<double, TestMemorySpace> interpolator,
             Kokkos::View<int *, TestMemorySpace> isel,
             Rank2View<double, TestMemorySpace> splinv) {

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
  // int ier;

  // std::vector<double> fget_vec(10);
  Kokkos::View<double *, TestMemorySpace> fget_vec("fget_vec",
                                                   10 * ntest * ntest);

  auto splinv_reshaped = Rank3View<double, TestMemorySpace>(
      splinv.data_handle(), ntest, ntest, 10);

  //   double zth = 0.0;
  //   double zx = 0.0;
  //   double ff = 0.0;
  //   for (int j = 0; j < ntest; ++j) {
  //     zth = thtest[j];
  //     for (int i = 0; i < ntest; ++i) {
  //       zx = xtest[i];
  //       ff = fxtest[i] * fthtest[j];
  //       fmin = std::min(fmin, ff);
  //       fmax = std::max(fmax, ff);
  //       int ier = 0;
  //       auto fget = Rank1View<double, TestMemorySpace>(fget_vec.data() + (j *
  //       ntest + i) * 10, 10);

  //       if (iherm == 0) {
  //         ExplicitBiCubicSplineInterpolator<double,
  //         TestMemorySpace>::bcspeval(zx, zth, isel, fget, x, nx, th, nth, f,
  //         ier);
  //       } else if (iherm == 2) {
  //         CompactBiCubicSplineInterpolator<double,
  //         TestMemorySpace>::evbicub(zx, zth, isel, fget, x, nx, th, nth, fh,
  //         ier);
  //       }

  //       if (ier == 0) {
  //         double fs = fget(0); // Interpolated value
  //         fdif = std::max(fdif, std::abs(ff - fs));
  //         fdifr = std::max(fdifr, std::abs((ff - fs) / (0.5 * (ff + fs))));
  //       }

  //     }
  //   }
  double result_fdif = 0.0;
  double result_fdifr = 0.0;
  Kokkos::parallel_reduce(
      "bicubic_eval", ntest * ntest,
      KOKKOS_LAMBDA(const int idx, double &max_fdif, double &max_fdifr) {
        int ier = 0;
        int j = idx / ntest;
        int i = idx % ntest;
        double zth = thtest(j);
        double zx = xtest(i);
        double ff = fxtest(i) * fthtest(j);
        auto fget =
            Rank1View<double, TestMemorySpace>(fget_vec.data() + idx * 10, 10);
        if (iherm == 0) {
          ExplicitBiCubicSplineInterpolator<double, TestMemorySpace>::bcspeval(
              zx, zth, isel, fget, x, nx, th, nth, f, ier);
        } else if (iherm == 2) {
          CompactBiCubicSplineInterpolator<double, TestMemorySpace>::evbicub(
              zx, zth, isel, fget, x, nx, th, nth, fh, ier);
        }
        if (ier == 0) {
          double fs = fget(0); // Interpolated value
          double dif = std::abs(ff - fs);
          max_fdif = Kokkos::max(max_fdif, dif);
          max_fdifr = Kokkos::max(max_fdifr, dif / (0.5 * (ff + fs)));
        }
        // printf("i=%d, j=%d,ff=%f, fs=%f, fdif=%f, fdifr=%f\n",
        //        i, j, ff, fget(0), max_fdif, max_fdifr);
      },
      Kokkos::Max<double>(result_fdif), Kokkos::Max<double>(result_fdifr));
  fdif = std::max(fdif, result_fdif);
  fdifr = std::max(fdifr, result_fdifr);

  std::cout << "2d" << slbl << "  min: " << fmin << "  max: " << fmax
            << "  dif: " << fdif << "  difr: " << fdifr << std::endl;
  assert(are_equal(fdif, 1.8312E-03));
  assert(are_equal(fdifr, 6.7151E-04));
}

void dotest2(Rank1View<double, TestMemorySpace> x,
             Rank1View<double, TestMemorySpace> fx, int nx,
             Rank1View<double, TestMemorySpace> th,
             Rank1View<double, TestMemorySpace> fth,
             Rank1View<double, TestMemorySpace> dfth, int nth,
             Rank4View<double, TestMemorySpace> f,
             Rank3View<double, TestMemorySpace> fh,
             Rank2View<double, TestMemorySpace> flin,
             Rank1View<double, TestMemorySpace> bcx1,
             Rank1View<double, TestMemorySpace> bcx2,
             Rank1View<double, TestMemorySpace> bcth1,
             Rank1View<double, TestMemorySpace> bcth2,
             Rank1View<double, TestMemorySpace> xtest,
             Rank1View<double, TestMemorySpace> fxtest,
             Rank1View<double, TestMemorySpace> thtest,
             Rank1View<double, TestMemorySpace> fthtest, int ntest) {
  // for (int ith = 0; ith < nth; ++ith) {
  //   for (int ix = 0; ix < nx; ++ix) {
  //     flin(ix, ith) = f(0, 0, ix, ith);          // f
  //     fh(0, ix, ith) = f(0, 0, ix, ith);         // f
  //     fh(1, ix, ith) = 2.0 * f(0, 0, ix, ith);   // df/dx
  //     fh(2, ix, ith) = fx[ix] * dfth[ith];       // df/dy
  //     fh(3, ix, ith) = 2.0 * fx[ix] * dfth[ith]; // d2f/dxdy
  //   }
  // }
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nth, nx}),
      KOKKOS_LAMBDA(const int ith, const int ix) {
        flin(ix, ith) = f(0, 0, ix, ith);          // f
        fh(0, ix, ith) = f(0, 0, ix, ith);         // f
        fh(1, ix, ith) = 2.0 * f(0, 0, ix, ith);   // df/dx
        fh(2, ix, ith) = fx(ix) * dfth(ith);       // df/dy
        fh(3, ix, ith) = 2.0 * fx(ix) * dfth(ith); // d2f/dxdy
      });

  int ier = 0;
  int nbc = 1;
  int ilinx = 0;
  int ilinth = 0;
  Kokkos::View<double *, TestMemorySpace> wk_vec("wk_vec", 1000);
  auto wk = Rank1View<double, TestMemorySpace>(wk_vec.data(), 1000);

  // BiCubicSplineInterpolator<double, TestMemorySpace> interpolator;

  bset(fx, nx, fth, nth, bcx1, bcx2, bcth1, bcth2);

  // interpolator.bcspline(x, nx, th, nth, f, nbc, bcx1, nbc, bcx2, nbc, bcth1,
  //                       nbc, bcth2, wk);
  ExplicitBiCubicSplineInterpolator<double, TestMemorySpace>
      explicit_interpolator(x, nx, th, nth, f, nbc, bcx1, nbc, bcx2, nbc, bcth1,
                            nbc, bcth2, wk);
  if (ier != 0) {
    std::cerr << " ?? error in pspltest:  dotest2(bcspline)\n";
    return;
  }
  Kokkos::View<double *, TestMemorySpace> splinv_view("splinv_view", 400000);
  auto splinv =
      Rank2View<double, TestMemorySpace>(splinv_view.data(), 40000, 10);
  int isel_arr[10] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Kokkos::View<int *, TestMemorySpace> isel("isel", 10);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TestMemorySpace>(0, isel.size()),
      KOKKOS_LAMBDA(const int i) { isel(i) = isel_arr[i]; });

  Kokkos::View<double *, TestMemorySpace> xtest_grid_view("xtest_grid_view",
                                                          40000);
  Kokkos::View<double *, TestMemorySpace> thtest_grid_view("thtest_grid_view",
                                                           40000);
  createFlatGrid(xtest, thtest, xtest_grid_view, thtest_grid_view);
  auto xtest_grid =
      Rank1View<double, TestMemorySpace>(xtest_grid_view.data(), 40000);
  auto thtest_grid =
      Rank1View<double, TestMemorySpace>(thtest_grid_view.data(), 40000);

  // TODO: evaluate function raise memeory acess violation
  //  explicit_interpolator.evaluate_explicit(isel, xtest_grid, thtest_grid,
  //  splinv);

  compare("bcspline", x, nx, th, nth, f, fh, flin, ilinx, ilinth, xtest, fxtest,
          thtest, fthtest, ntest, explicit_interpolator, isel, splinv);

  // for (int ith = 0; ith < nth; ++ith) {
  //   for (int ix = 0; ix < nx; ++ix) {
  //     fh(0, ix, ith) = f(0, 0, ix, ith);
  //   }
  // }

  // interpolator.mkbicub(x, nx, th, nth, fh, nbc, bcx1, nbc, bcx2, nbc, bcth1,
  //                      nbc, bcth2, wk);
  CompactBiCubicSplineInterpolator<double, TestMemorySpace>
      compact_interpolator(x, nx, th, nth, fh, nbc, bcx1, nbc, bcx2, nbc, bcth1,
                           nbc, bcth2, wk);
  //   compact_interpolator.evaluate_compact(isel, xtest_grid, thtest_grid,
  //   splinv);

  compare("mkbicub", x, nx, th, nth, f, fh, flin, ilinx, ilinth, xtest, fxtest,
          thtest, fthtest, ntest, compact_interpolator, isel, splinv);
}

void pspltest2(double zctrl) {
  const double pi2 = 6.28318530718;
  const double zero = 0.0;
  const double one = 1.0;

  Kokkos::View<double *, TestMemorySpace> x1_view("x1_view", 10);
  Kokkos::View<double *, TestMemorySpace> ex1_view("ex1_view", 10);
  Kokkos::View<double *, TestMemorySpace> t1_view("t1_view", 10);
  Kokkos::View<double *, TestMemorySpace> st1_view("st1_view", 10);
  Kokkos::View<double *, TestMemorySpace> ct1_view("ct1_view", 10);
  Kokkos::View<double *, TestMemorySpace> xtest_view("xtest_view", 200);
  Kokkos::View<double *, TestMemorySpace> extest_view("extest_view", 200);
  Kokkos::View<double *, TestMemorySpace> ttest_view("ttest_view", 200);
  Kokkos::View<double *, TestMemorySpace> stest_view("stest_view", 200);
  Kokkos::View<double *, TestMemorySpace> ctest_view("ctest_view", 200);
  Kokkos::View<double *, TestMemorySpace> f1_view("f1_view", 1600);
  Kokkos::View<double *, TestMemorySpace> f2_view("f2_view", 6400);
  Kokkos::View<double *, TestMemorySpace> f4_view("f4_view", 4 * 4 * 40 * 40);
  Kokkos::View<double *, TestMemorySpace> bcx1_view("bcx1_view", 40);
  Kokkos::View<double *, TestMemorySpace> bcx2_view("bcx2_view", 40);
  Kokkos::View<double *, TestMemorySpace> bcth1_view("bcth1_view", 40);
  Kokkos::View<double *, TestMemorySpace> bcth2_view("bcth2_view", 40);
  Kokkos::View<double *, TestMemorySpace> fh_view("fh_view", 400);
  Kokkos::View<double *, TestMemorySpace> flin_view("flin_view", 100);

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

  auto x1 = Rank1View<double, TestMemorySpace>(x1_view.data(), x1_view.size());
  auto ex1 =
      Rank1View<double, TestMemorySpace>(ex1_view.data(), ex1_view.size());
  auto t1 = Rank1View<double, TestMemorySpace>(t1_view.data(), t1_view.size());
  auto st1 =
      Rank1View<double, TestMemorySpace>(st1_view.data(), st1_view.size());
  auto ct1 =
      Rank1View<double, TestMemorySpace>(ct1_view.data(), ct1_view.size());
  auto f1 = Rank4View<double, TestMemorySpace>(f1_view.data(),
                                               Kokkos::extents{4, 4, 10, 10});
  auto flin = Rank2View<double, TestMemorySpace>(flin_view.data(), 10, 10);

  auto fh = Rank3View<double, TestMemorySpace>(fh_view.data(), 4, 10, 10);

  auto bcx1 =
      Rank1View<double, TestMemorySpace>(bcx1_view.data(), bcx1_view.size());
  auto bcx2 =
      Rank1View<double, TestMemorySpace>(bcx2_view.data(), bcx2_view.size());
  auto bcth1 =
      Rank1View<double, TestMemorySpace>(bcth1_view.data(), bcth1_view.size());
  auto bcth2 =
      Rank1View<double, TestMemorySpace>(bcth2_view.data(), bcth2_view.size());

  auto xtest =
      Rank1View<double, TestMemorySpace>(xtest_view.data(), xtest_view.size());
  auto extest = Rank1View<double, TestMemorySpace>(extest_view.data(),
                                                   extest_view.size());
  auto ttest =
      Rank1View<double, TestMemorySpace>(ttest_view.data(), ttest_view.size());
  auto stest =
      Rank1View<double, TestMemorySpace>(stest_view.data(), stest_view.size());

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
  std::cout << "DefaultExecutionSpace: "
            << Kokkos::DefaultExecutionSpace::name() << std::endl;
  Kokkos::Timer timer;
  pspltest1(0.01);
  pspltest2(0.01);
  double time = timer.seconds();
  std::cout << "Total time for spline tests: " << time << " seconds"
            << std::endl;
  Kokkos::finalize();
  MPI_Finalize();
}