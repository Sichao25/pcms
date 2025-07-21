
#ifndef MLS_RBF_OPTIONS_HPP
#define MLS_RBF_OPTIONS_HPP

#include "mdspan/mdspan.hpp"
#include "pcms/arrays.h"
#include "pcms/memory_spaces.h"
#include "pcms/types.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include "pcms/assert.h"

namespace pcms {

// Todo: remove this once target PR is merged
// Declaration of customized Kokkos types
template <typename ElementType, typename MemorySpace>
struct memory_space_accessor : public Kokkos::default_accessor<ElementType> {
  using memory_space = MemorySpace;
};

template <int Rank, typename ElementType, typename MemorySpace>
using View = Kokkos::mdspan<
    ElementType, Kokkos::dextents<LO, Rank>, Kokkos::layout_right,
    detail::memory_space_accessor<std::remove_reference_t<ElementType>,
                                  MemorySpace>>;

template <typename ElementType, typename MemorySpace>
using Rank1View = View<1, ElementType, MemorySpace>;

template <typename ElementType, typename MemorySpace>
using Rank2View = View<2, ElementType, MemorySpace>;

template <typename ElementType, typename MemorySpace>
using Rank3View = View<3, ElementType, MemorySpace>;

template <typename ElementType, typename MemorySpace>
using Rank4View = View<4, ElementType, MemorySpace>;

template <typename T, typename MemorySpace> class SplineInterpolator {
public:
  SplineInterpolator() = default;
  void splinck(Rank1View<T, MemorySpace> x, const double &ztol);
  using execution_space = typename MemorySpace::execution_space;
};

template <typename T, typename MemorySpace>
void SplineInterpolator<T, MemorySpace>::splinck(
    Rank1View<T, MemorySpace> x, const double &ztol) {
  int inx = static_cast<int>(x.extent(0));

  if (inx <= 1)
    return;

  double dxavg = (x[inx - 1] - x[0]) / (inx - 1);
  double zeps = std::abs(ztol * dxavg);

  for (int ix = 1; ix < inx; ++ix) {
    double zdiffx = x[ix] - x[ix - 1];
    if (zdiffx <= 0.0) {
      std::ostringstream msg;
      msg << "Array not strictly ascending: x[" << ix << "] = " << x[ix] 
          << " <= x[" << (ix-1) << "] = " << x[ix-1] 
          << " (difference = " << zdiffx << ")";
      throw std::runtime_error(msg.str());
    }
    double zdiff = zdiffx - dxavg;
    if (std::abs(zdiff) > zeps) {
      std::ostringstream msg;
      msg << "Non-uniform grid detected: spacing at index " << ix 
          << " is " << zdiffx << ", expected " << dxavg 
          << " (tolerance: " << zeps << ", difference: " << std::abs(zdiff) << ")";
      throw std::runtime_error(msg.str());
    }
  }
}

// TODO: better error handling, use pcms existing error handling
// TODO: better documentation
// TODO: add more checks/assetions for input parameters
// TODO: remove the usage of size variable if it is better to use extent(0) or
// extent(1) in mdspan
template <typename T, typename MemorySpace>
class CubicSplineInterpolator : public SplineInterpolator<T, MemorySpace> {
public:
  CubicSplineInterpolator() = default;

  CubicSplineInterpolator(
    Rank1View<T, MemorySpace> x, const int &nx,
    Rank2View<T, MemorySpace> fspl, const int &ibcxmin, const double &bcxmin,
    const int &ibcxmax, const double &bcxmax, Rank1View<T, MemorySpace> wk);
  
  CubicSplineInterpolator(
    Rank1View<T, MemorySpace> x, const int &nx,
    Rank2View<T, MemorySpace> fspl, Rank2View<T, MemorySpace> fspl4,
    const int &ibcxmin, const double &bcxmin, const int &ibcxmax,
    const double &bcxmax, Rank1View<T, MemorySpace> wk);

  using typename SplineInterpolator<T, MemorySpace>::execution_space;

  Rank2View<T, MemorySpace> fspl_;
  Rank2View<T, MemorySpace> fs2_;
  Rank1View<T, MemorySpace> x_;
  int nx_;

  void cspline(Rank1View<T, MemorySpace> x, const int &nx,
               Rank2View<T, MemorySpace> fspl, const int &ibcxmin,
               const double &bcxmin, const int &ibcxmax, const double &bcxmax,
               Rank1View<T, MemorySpace> wk);

  KOKKOS_FUNCTION void v_spline(const int &k_bc1, const int &k_bcn, const int &n,
                Rank1View<T, MemorySpace> x, Rank2View<T, MemorySpace> f,
                Rank1View<T, MemorySpace> wk);

  KOKKOS_FUNCTION void cspevfn(Kokkos::View<int*, MemorySpace> selector, Rank1View<T, MemorySpace> fval, const int &i,
               const double &dx);

  void mkspline(Rank1View<T, MemorySpace> x, const int &nx,
                Rank2View<T, MemorySpace> fspl, Rank2View<T, MemorySpace> fspl4,
                const int &ibcxmin, const double &bcxmin, const int &ibcxmax,
                const double &bcxmax, Rank1View<T, MemorySpace> wk);

  KOKKOS_FUNCTION void fvspline(Kokkos::View<int*, MemorySpace> selector,
                Rank1View<T, MemorySpace> fval, const int &i,
                const double &xparam, const double &hx, const double &hxi);

  void cspeval(double xget, Kokkos::View<int*, MemorySpace> iselect,
               Rank1View<T, MemorySpace> fval,
               int &ier);

  KOKKOS_FUNCTION void cspevx(double xget, Rank1View<T, MemorySpace> x, const int &nx,
              int &i, double &dx, int &ier);

  void evspline(double xget, Kokkos::View<int*, MemorySpace> ict,
                Rank1View<T, MemorySpace> fval,
                int &ier);

  KOKKOS_FUNCTION void herm1x(double xget, Rank1View<T, MemorySpace> x, const int &nx,
              int &i, double &xparam, double &hx, double &hxi, int &ier);
  void evaluate_explicit(
    Kokkos::View<int*, MemorySpace> selector,
    Rank1View<T, MemorySpace> xvec,
    Rank2View<T, MemorySpace> fval);
  
  void evaluate_compact(
    Kokkos::View<int*, MemorySpace> selector,
    Rank1View<T, MemorySpace> xvec,
    Rank2View<T, MemorySpace> fval);
}; // end of cubic spline class

template <typename T, typename MemorySpace>
CubicSplineInterpolator<T, MemorySpace>::CubicSplineInterpolator(
    Rank1View<T, MemorySpace> x, const int &nx,
    Rank2View<T, MemorySpace> fspl, const int &ibcxmin, const double &bcxmin,
    const int &ibcxmax, const double &bcxmax, Rank1View<T, MemorySpace> wk) {
      nx_ = x.extent(0);
      x_ = x;
      cspline(x, nx, fspl, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);
      fspl_ = fspl;
}


template <typename T, typename MemorySpace>
CubicSplineInterpolator<T, MemorySpace>::CubicSplineInterpolator(
    Rank1View<T, MemorySpace> x, const int &nx,
    Rank2View<T, MemorySpace> fspl, Rank2View<T, MemorySpace> fspl4,
    const int &ibcxmin, const double &bcxmin, const int &ibcxmax,
    const double &bcxmax, Rank1View<T, MemorySpace> wk) {
      nx_ = x.extent(0);
      x_ = x;
      mkspline(x, nx, fspl, fspl4, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);
      fs2_ = fspl;
    }

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::cspline(
    Rank1View<T, MemorySpace> x, const int &nx,
    Rank2View<T, MemorySpace> fspl, const int &ibcxmin, const double &bcxmin,
    const int &ibcxmax, const double &bcxmax, Rank1View<T, MemorySpace> wk) {
  if (x.extent(0) < 2) {
    throw std::invalid_argument(
        "Cubic spline requires at least 2 points in each dimension.");
  }
  // TODO: check min max between -1, 0 - 7 ?

  x_ = x;
  nx_ = x.extent(0);

  double half = 0.5;
  double sixth = 0.166666666666666667;

  if (ibcxmin == 1) {
    fspl(1, 0) = bcxmin;
  } else if (ibcxmin == 2) {
    fspl(2, 0) = bcxmin;
  }

  if (ibcxmax == 1) {
    fspl(1, nx - 1) = bcxmax;
  } else if (ibcxmax == 2) {
    fspl(2, nx - 1) = bcxmax;
  }
  v_spline(ibcxmin, ibcxmax, nx, x, fspl, wk);
  // Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, nx),
  // KOKKOS_LAMBDA (const int i) {
  //     fspl(2, i) = half * fspl(2, i);
  //     fspl(3, i) = sixth * fspl(3, i);
  // });
  for (int i = 0; i < nx; ++i) {
    fspl(2, i) = half * fspl(2, i);
    fspl(3, i) = sixth * fspl(3, i);
  }

  fspl_ = fspl;
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void CubicSplineInterpolator<T, MemorySpace>::v_spline(
    const int &k_bc1, const int &k_bcn, const int &n,
    Rank1View<T, MemorySpace> x, Rank2View<T, MemorySpace> f,
    Rank1View<T, MemorySpace> wk) {
  int i_bc1 = k_bc1;
  int i_bcn = k_bcn;
  int iord1, iord2, imin, imax;
  double a1, b1, an, bn, f0, fh, h;

  // Clip to allowed ranges
  if (i_bc1 < -1 || i_bc1 > 7)
    i_bc1 = 0; // outside [-1,7] -> not-a-knot
  if (i_bcn < 0 || i_bcn > 7)
    i_bcn = 0; // outside [0,7]  -> not-a-knot

  // Periodic BC handling
  if (i_bc1 == -1)
    i_bcn = -1;

  int i3knots = 0;
  int i3perio = 0;
  // TODO: revisit BC check
  if (n == 3) {
    i_bc1 = std::min(6, i_bc1);
    i_bcn = std::min(6, i_bcn);

    if (i_bc1 == 0)
      i3knots += 1;
    if (i_bcn == 0)
      i3knots += 1;
    if (i_bc1 == -1)
      i3perio = 1;
  }

  if (n == 2) {
    if (i_bc1 == -1) {
      i_bc1 = 5;
      i_bcn = 5;
    }

    if (i_bc1 == 0 || i_bc1 > 5)
      i_bc1 = 5;
    if (i_bcn == 0 || i_bcn > 5)
      i_bcn = 5;

    // LHS match
    if (i_bc1 == 1 || i_bc1 == 3 || i_bc1 == 5)
      iord1 = 1; // first derivative match
    else
      iord1 = 2; // second derivative match
    // RHS match
    if (i_bcn == 1 || i_bcn == 3 || i_bcn == 5)
      iord2 = 1;
    else
      iord2 = 2;
  }
  imin = 0;
  imax = n - 1;

  a1 = 0.0;
  b1 = 0.0;
  an = 0.0;
  bn = 0.0;

  if (i_bc1 == 1) {
    a1 = f(1, 0);
  } else if (i_bc1 == 2) {
    b1 = f(2, 0);
  } else if (i_bc1 == 5) {
    a1 = (f(0, 1) - f(0, 0)) / (x[1] - x[0]);
  } else if (i_bc1 == 6) {
    b1 = 2.0 *
         ((f(0, 2) - f(0, 1)) / (x[2] - x[1]) -
          (f(0, 1) - f(0, 0)) / (x[1] - x[0])) /
         (x[2] - x[0]);
  }

  if (i_bcn == 1) {
    an = f(1, n - 1);
  } else if (i_bcn == 2) {
    bn = f(2, n - 1);
  } else if (i_bcn == 5) {
    an = (f(0, n - 1) - f(0, n - 2)) / (x[n - 1] - x[n - 2]);
  } else if (i_bcn == 6) {
    bn = 2.0 *
         ((f(0, n - 1) - f(0, n - 2)) / (x[n - 1] - x[n - 2]) -
          (f(0, n - 2) - f(0, n - 3)) / (x[n - 2] - x[n - 3])) /
         (x[n - 1] - x[n - 3]);
  }
  f(1, n - 1) = 0.0;
  f(2, n - 1) = 0.0;
  f(3, n - 1) = 0.0;
  if (n == 2) {
    if (i_bc1 == 5 && i_bcn == 5) {
      // Coefficients for n = 2
      f(1, 0) = (f(0, 1) - f(0, 0)) / (x[1] - x[0]);
      f(2, 0) = 0.0;
      f(3, 0) = 0.0;
      f(1, 1) = f(1, 0);
      f(2, 1) = 0.0;
      f(3, 1) = 0.0;
    } else if (iord1 == 1 && iord2 == 1) {
      double a1 = f(1, 0), an = f(1, 1);
      f(1, 0) = a1;
      f(1, 1) = an;
      h = x[1] - x[0];
      f0 = f(0, 0);
      fh = f(0, 1);

      f(2, 0) = (3 * (fh - f0) / (h * h) - (2 * a1 + an) / h) * 2; // 2*c2
      f(3, 0) =
          (-2 * (fh - f0) / (h * h * h) + (a1 + an) / (h * h)) * 6; // 6*c1

      f(3, 1) = f(3, 0);
      f(2, 1) = f(3, 0) * h + f(2, 0);

    } else if (iord1 == 1 && iord2 == 2) {
      double a1 = f(1, 0), bn = f(2, 1);
      f(1, 0) = a1;
      f(2, 1) = bn;
      h = x[1] - x[0];
      f0 = f(0, 0);
      fh = f(0, 1);

      f(2, 0) = (-bn / 4 + 3 * (fh - f0) / (2 * h * h) - 3 * a1 / (2 * h)) * 2;
      f(3, 0) =
          (bn / (4 * h) - (fh - f0) / (2 * h * h * h) + a1 / (2 * h * h)) * 6;

      f(3, 1) = f(3, 0);
      f(1, 1) = f(3, 0) * h * h / 2 + f(2, 0) * h + a1;

    } else if (iord1 == 2 && iord2 == 1) {
      double b1 = f(2, 0), an = f(1, 1);
      f(2, 0) = b1;
      f(1, 1) = an;
      h = x[1] - x[0];
      f0 = f(0, 0);
      fh = f(0, 1);

      f(1, 0) = 3 * (fh - f0) / (2 * h) - b1 * h / 4 - an / 2; // c3
      f(3, 0) =
          (an / (2 * h * h) - (fh - f0) / (2 * h * h * h) - b1 / (4 * h)) * 6;

      f(3, 1) = f(3, 0);
      f(2, 1) = f(3, 0) * h + f(2, 0);

    } else if (iord1 == 2 && iord2 == 2) {
      double b1 = f(2, 0), bn = f(2, 1);
      f(2, 0) = b1;
      f(2, 1) = bn;
      h = x[1] - x[0];
      f0 = f(0, 0);
      fh = f(0, 1);

      f(1, 0) = (fh - f0) / h - b1 * h / 3 - bn * h / 6;
      f(3, 0) = (bn - b1) / h;

      f(3, 1) = f(3, 0);
      f(1, 1) = f(3, 0) * h * h / 2 + b1 * h + f(1, 0);
    }
  } // n == 2
  else if (i3perio == 1) {
    double h1 = x[1] - x[0];
    double h2 = x[2] - x[1];
    double h = h1 + h2;

    double dels = (f(0, 2) - f(0, 1)) / h2 - (f(0, 1) - f(0, 0)) / h1;

    f(1, 0) = (f(0, 1) - f(0, 0)) / h1 + (h1 * dels) / h;
    f(2, 0) = -6.0 * dels / h;
    f(3, 0) = 12.0 * dels / (h1 * h);

    f(1, 1) = (f(0, 2) - f(0, 1)) / h2 - (h2 * dels) / h;
    f(2, 1) = 6.0 * dels / h;
    f(3, 1) = -12.0 * dels / (h2 * h);

    f(1, 2) = f(1, 0);
    f(2, 2) = f(2, 0);
    f(3, 2) = f(3, 0);
  } // i3perio == 1
  else if (i3knots == 2) {
    // Special case: nx = 3, not-a-knot on both sides
    double h1 = x[1] - x[0];
    double h2 = x[2] - x[1];
    double h = h1 + h2;

    double f1 = f(0, 0) - f(0, 1);
    double f2 = f(0, 2) - f(0, 1);

    // Solve quadratic through 3 points centered at x[1]
    double aa = (f2 * h1 + f1 * h2) / (h1 * h2 * h);
    double bb = (f2 * h1 * h1 - f1 * h2 * h2) / (h1 * h2 * h);

    // Third derivative = 0 (quadratic)
    f(3, 0) = 0.0;
    f(3, 1) = 0.0;
    f(3, 2) = 0.0;

    // Second derivative is constant
    f(2, 0) = 2 * aa;
    f(2, 1) = 2 * aa;
    f(2, 2) = 2 * aa;

    // First derivatives
    f(1, 0) = bb - 2 * aa * h1;
    f(1, 1) = bb;
    f(1, 2) = bb + 2 * aa * h2;
  } // i3knots == 2
  else if (i3knots == 1) {
    if (i_bc1 == 1 || i_bc1 == 3 || i_bc1 == 5) {
      // f' LHS condition; not-a-knot RHS
      double h2 = x[1] - x[0];
      double h3 = x[2] - x[0];

      double f2 = f(0, 1) - f(0, 0);
      double f3 = f(0, 2) - f(0, 0);

      double aa = a1 / (h2 * h3) + f3 / (h3 * h3 * (h3 - h2)) -
                  f2 / (h2 * h2 * (h3 - h2));
      double bb = -a1 * (h3 * h3 - h2 * h2) / (h2 * h3 * (h3 - h2)) +
                  f2 * h3 / (h2 * h2 * (h3 - h2)) -
                  f3 * h2 / (h3 * h3 * (h3 - h2));

      f(1, 0) = a1;
      f(2, 0) = 2 * bb;
      f(3, 0) = 6 * aa;

      f(1, 1) = 3 * aa * h2 * h2 + 2 * bb * h2 + a1;
      f(2, 1) = 6 * aa * h2 + 2 * bb;
      f(3, 1) = 6 * aa;

      f(1, 2) = 3 * aa * h3 * h3 + 2 * bb * h3 + a1;
      f(2, 2) = 6 * aa * h3 + 2 * bb;
      f(3, 2) = 6 * aa;

    } else if (i_bc1 == 2 || i_bc1 == 4 || i_bc1 == 6) {
      // f'' LHS condition; not-a-knot RHS
      double h2 = x[1] - x[0];
      double h3 = x[2] - x[0];

      double f2 = f(0, 1) - f(0, 0);
      double f3 = f(0, 2) - f(0, 0);

      double aa = -(b1 / 2.0) * (h3 - h2) / (h3 * h3 - h2 * h2) -
                  f2 / (h2 * (h3 * h3 - h2 * h2)) +
                  f3 / (h3 * (h3 * h3 - h2 * h2));
      double bb = -(b1 / 2.0) * h2 * h3 * (h3 - h2) / (h3 * h3 - h2 * h2) +
                  f2 * h3 * h3 / (h2 * (h3 * h3 - h2 * h2)) -
                  f3 * h2 * h2 / (h3 * (h3 * h3 - h2 * h2));

      f(1, 0) = bb;
      f(2, 0) = b1;
      f(3, 0) = 6 * aa;

      f(1, 1) = 3 * aa * h2 * h2 + b1 * h2 + bb;
      f(2, 1) = 6 * aa * h2 + b1;
      f(3, 1) = 6 * aa;

      f(1, 2) = 3 * aa * h3 * h3 + b1 * h3 + bb;
      f(2, 2) = 6 * aa * h3 + b1;
      f(3, 2) = 6 * aa;

    } else if (i_bcn == 1 || i_bcn == 3 || i_bcn == 5) {
      // f' RHS condition; not-a-knot LHS
      double h2 = x[1] - x[2];
      double h3 = x[0] - x[2];

      double f2 = f(0, 1) - f(0, 2);
      double f3 = f(0, 0) - f(0, 2);

      double aa = an / (h2 * h3) + f3 / (h3 * h3 * (h3 - h2)) -
                  f2 / (h2 * h2 * (h3 - h2));
      double bb = -an * (h3 * h3 - h2 * h2) / (h2 * h3 * (h3 - h2)) +
                  f2 * h3 / (h2 * h2 * (h3 - h2)) -
                  f3 * h2 / (h3 * h3 * (h3 - h2));

      f(1, 2) = an;
      f(2, 2) = 2 * bb;
      f(3, 2) = 6 * aa;

      f(1, 1) = 3 * aa * h2 * h2 + 2 * bb * h2 + an;
      f(2, 1) = 6 * aa * h2 + 2 * bb;
      f(3, 1) = 6 * aa;

      f(1, 0) = 3 * aa * h3 * h3 + 2 * bb * h3 + an;
      f(2, 0) = 6 * aa * h3 + 2 * bb;
      f(3, 0) = 6 * aa;

    } else if (i_bcn == 2 || i_bcn == 4 || i_bcn == 6) {
      // f'' RHS condition; not-a-knot LHS
      double h2 = x[1] - x[2];
      double h3 = x[0] - x[2];

      double f2 = f(0, 1) - f(0, 2);
      double f3 = f(0, 0) - f(0, 2);

      double aa = -(bn / 2.0) * (h3 - h2) / (h3 * h3 - h2 * h2) -
                  f2 / (h2 * (h3 * h3 - h2 * h2)) +
                  f3 / (h3 * (h3 * h3 - h2 * h2));
      double bb = -(bn / 2.0) * h2 * h3 * (h3 - h2) / (h3 * h3 - h2 * h2) +
                  f2 * h3 * h3 / (h2 * (h3 * h3 - h2 * h2)) -
                  f3 * h2 * h2 / (h3 * (h3 * h3 - h2 * h2));

      f(1, 2) = bb;
      f(2, 2) = bn;
      f(3, 2) = 6 * aa;

      f(1, 1) = 3 * aa * h2 * h2 + bn * h2 + bb;
      f(2, 1) = 6 * aa * h2 + bn;
      f(3, 1) = 6 * aa;

      f(1, 0) = 3 * aa * h3 * h3 + bn * h3 + bb;
      f(2, 0) = 6 * aa * h3 + bn;
      f(3, 0) = 6 * aa;
    }
  } // i3knots == 1
  else if (n > 2) {
    f(3, 0) = x[1] - x[0];
    f(2, 1) = (f(0, 1) - f(0, 0)) / f(3, 0);

    // TODO: check if Kokkos do improves performance
    for (int i = 1; i < n - 1; ++i) {
      f(3, i) = x[i + 1] - x[i];
      f(1, i) = 2.0 * (f(3, i - 1) + f(3, i));
      f(2, i + 1) = (f(0, i + 1) - f(0, i)) / f(3, i);
      f(2, i) = f(2, i + 1) - f(2, i);
    }
    // Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(1, n - 1),
    // KOKKOS_LAMBDA (const int i) {
    //     f(3, i) = x[i + 1] - x[i];
    // });
    // Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(1, n - 1),
    // KOKKOS_LAMBDA (const int i) {
    //     f(1, i) = 2.0 * (f(3, i - 1) + f(3, i));
    //     f(2, i + 1) = (f(0, i + 1) - f(0, i)) / f(3, i);
    // });
    // for (int i = 1; i < n - 1; ++i) {
    //     f(2, i) = f(2, i + 1) - f(2, i);
    // }

    double elem21 = f(3, 0);
    double elemnn1 = f(3, n - 2);

    // Left boundary conditions
    if (i_bc1 == -1) {
      f(1, 0) = 2.0 * (f(3, 0) + f(3, n - 2));
      f(2, 0) = (f(0, 1) - f(0, 0)) / f(3, 0) -
                (f(0, n - 1) - f(0, n - 2)) / f(3, n - 2);
      wk[0] = f(3, n - 2);
      for (int i = 1; i <= n - 4; ++i)
        wk[i] = 0.0;
      wk[n - 3] = f(3, n - 3);
      wk[n - 2] = f(3, n - 2);
    } else if (i_bc1 == 1 || i_bc1 == 3 || i_bc1 == 5) {
      f(1, 0) = 2.0 * f(3, 0);
      f(2, 0) = (f(0, 1) - f(0, 0)) / f(3, 0) - a1;
    } else if (i_bc1 == 2 || i_bc1 == 4 || i_bc1 == 6) {
      f(1, 0) = 2.0 * f(3, 0);
      f(2, 0) = f(3, 0) * b1 / 3.0;
      f(3, 0) = 0.0;
    } else if (i_bc1 == 7) {
      f(1, 0) = -f(3, 0);
      f(2, 0) = f(2, 2) / (x[3] - x[1]) - f(2, 1) / (x[2] - x[0]);
      f(2, 0) *= f(3, 0) * f(3, 0) / (x[3] - x[0]);
    } else {
      imin = 1;
      f(1, 1) = f(3, 0) + 2.0 * f(3, 1);
      f(2, 1) = f(2, 1) * f(3, 1) / (f(3, 0) + f(3, 1));
    }

    // Right boundary conditions
    if (i_bcn == 1 || i_bcn == 3 || i_bcn == 5) {
      f(1, n - 1) = 2.0 * f(3, n - 2);
      f(2, n - 1) = -(f(0, n - 1) - f(0, n - 2)) / f(3, n - 2) + an;
    } else if (i_bcn == 2 || i_bcn == 4 || i_bcn == 6) {
      f(1, n - 1) = 2.0 * f(3, n - 2);
      f(2, n - 1) = f(3, n - 2) * bn / 3.0;
      elemnn1 = 0.0;
    } else if (i_bcn == 7) {
      f(1, n - 1) = -f(3, n - 2);
      f(2, n - 1) = f(2, n - 2) / (x[n - 1] - x[n - 3]) -
                    f(2, n - 3) / (x[n - 2] - x[n - 4]);
      f(2, n - 1) =
          -f(2, n - 1) * f(3, n - 2) * f(3, n - 2) / (x[n - 1] - x[n - 4]);
    } else if (i_bc1 != -1) {
      imax = n - 2;
      f(1, n - 2) = 2.0 * f(3, n - 3) + f(3, n - 2);
      f(2, n - 2) = f(2, n - 2) * f(3, n - 3) / (f(3, n - 2) + f(3, n - 3));
    }
    if (i_bc1 == -1) {
      // Periodic Boundary Conditions

      // Forward elimination
      for (int i = 1; i <= n - 3; ++i) {
        double t = f(3, i - 1) / f(1, i - 1);
        f(1, i) -= t * f(3, i - 1);
        f(2, i) -= t * f(2, i - 1);
        wk[i] -= t * wk[i - 1];

        double q = wk[n - 2] / f(1, i - 1);
        wk[n - 2] = -q * f(3, i - 1);
        f(1, n - 2) -= q * wk[i - 1];
        f(2, n - 2) -= q * f(2, i - 1);
      }

      wk[n - 2] += f(3, n - 3);

      // Complete forward elimination
      double t = wk[n - 2] / f(1, n - 3);
      f(1, n - 2) -= t * wk[n - 3];
      f(2, n - 2) -= t * f(2, n - 3);

      // Back substitution
      f(2, n - 2) /= f(1, n - 2);
      f(2, n - 3) = (f(2, n - 3) - wk[n - 3] * f(2, n - 2)) / f(1, n - 3);
      for (int ib = 2; ib <= n - 2; ++ib) {
        int i = n - ib - 2;
        f(2, i) =
            (f(2, i) - f(3, i) * f(2, i + 1) - wk[i] * f(2, n - 2)) / f(1, i);
      }

      f(2, n - 1) = f(2, 0);

    } else {
      // Non-periodic Boundary Conditions

      for (int i = imin + 1; i <= imax; ++i) {
        double t;
        if ((i == n - 2) && (imax == n - 2)) {
          t = (f(3, i - 1) - f(3, i)) / f(1, i - 1);
        } else if (i == 1) {
          t = elem21 / f(1, i - 1);
        } else if (i == n - 1) {
          t = elemnn1 / f(1, i - 1);
        } else {
          t = f(3, i - 1) / f(1, i - 1);
        }

        if ((i == imin + 1) && (imin == 1)) {
          f(1, i) -= t * (f(3, i - 1) - f(3, i - 2));
        } else {
          f(1, i) -= t * f(3, i - 1);
        }

        f(2, i) -= t * f(2, i - 1);
      }

      f(2, imax) /= f(1, imax);
      for (int ib = 0; ib < imax - imin; ++ib) {
        int i = imax - 1 - ib;
        if ((i == 1) && (imin == 1)) {
          f(2, i) = (f(2, i) - (f(3, i) - f(3, i - 1)) * f(2, i + 1)) / f(1, i);
        } else {
          f(2, i) = (f(2, i) - f(3, i) * f(2, i + 1)) / f(1, i);
        }
      }

      f(3, 0) = x[1] - x[0];
      f(3, n - 2) = x[n - 1] - x[n - 2];

      if (i_bc1 <= 0 || i_bc1 > 7) {
        f(2, 0) = (f(2, 1) * (f(3, 0) + f(3, 1)) - f(2, 2) * f(3, 0)) / f(3, 1);
      }

      if (i_bcn <= 0 || i_bcn > 7) {
        f(2, n - 1) = f(2, n - 2) +
                      (f(2, n - 2) - f(2, n - 3)) * f(3, n - 2) / f(3, n - 3);
      }
    }

    // Polynomial coefficient computation
    for (int i = 0; i < n - 1; ++i) {
      f(1, i) = (f(0, i + 1) - f(0, i)) / f(3, i) -
                f(3, i) * (f(2, i + 1) + 2.0 * f(2, i));
      f(3, i) = (f(2, i + 1) - f(2, i)) / f(3, i);
      f(2, i) *= 6.0;
      f(3, i) *= 6.0;
    }
    // Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, n - 1),
    // KOKKOS_LAMBDA (const int i) {
    //     f(1, i) = (f(0, i + 1) - f(0, i)) / f(3, i) - f(3, i) * (f(2, i + 1)
    //     + 2.0 * f(2, i)); f(3, i) = (f(2, i + 1) - f(2, i)) / f(3, i);
    // });
    // Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, n - 1),
    // KOKKOS_LAMBDA (const int i) {
    //     f(2, i) *= 6.0;
    //     f(3, i) *= 6.0;
    // });

    if (i_bc1 == -1) {
      f(1, n - 1) = f(1, 0);
      f(2, n - 1) = f(2, 0);
      f(3, n - 1) = f(3, 0);
    } else {
      double hn = x[n - 1] - x[n - 2];
      f(1, n - 1) = f(1, n - 2) + hn * (f(2, n - 2) + 0.5 * hn * f(3, n - 2));
      f(2, n - 1) = f(2, n - 2) + hn * f(3, n - 2);
      f(3, n - 1) = f(3, n - 2);

      if (i_bcn == 1 || i_bcn == 3 || i_bcn == 5) {
        f(1, n - 1) = an;
      } else if (i_bcn == 2 || i_bcn == 4 || i_bcn == 6) {
        f(2, n - 1) = bn;
      }
    }
  }
} // v_spline

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void CubicSplineInterpolator<T, MemorySpace>::cspevfn(
    Kokkos::View<int*, MemorySpace> selector,
    Rank1View<T, MemorySpace> fval, const int &i, const double &dx) {

  int iaval = 0;

  if (selector[0] == 3) {
    // Third derivative only
    iaval++;
    fval(iaval - 1) = 6.0 * fspl_(3, i);
  } else {
    if (selector[0] > 0) {
      // Evaluate f
      iaval++;
      fval(iaval - 1) =
          fspl_(0, i) +
          dx * (fspl_(1, i) + dx * (fspl_(2, i) + dx * fspl_(3, i)));
    }

    if (selector[1] > 0) {
      // Evaluate df/dx
      iaval++;
      fval(iaval - 1) =
          fspl_(1, i) + dx * (2.0 * fspl_(2, i) + dx * 3.0 * fspl_(3, i));
    }

    if (selector[2] > 0) {
      // Evaluate d2f/dx2
      iaval++;
      fval(iaval - 1) = 2.0 * fspl_(2, i) + dx * 6.0 * fspl_(3, i);
    }
  }
} // end cspevfn


template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::cspeval(
    double xget, Kokkos::View<int*, MemorySpace> iselect,
    Rank1View<T, MemorySpace> fval,
    int &ier) {
  int ia = 0;
  double dxa = 0.0;
  cspevx(xget, x_, nx_, ia, dxa, ier);
  if (ier != 0) {
    printf("cspeval: error in cspevx, ier = %d\n", ier);
    return;
  }

  cspevfn(iselect, fval, ia, dxa);
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void CubicSplineInterpolator<T, MemorySpace>::cspevx(
    double xget, Rank1View<T, MemorySpace> x, const int &nx, int &i,
    double &dx, int &ier) {
  int nxm = nx - 1;
  double zxget = xget;

  // Range check
  if (xget < x[0] || xget > x[nx - 1]) {
    double zxtol = 4.0e-7 * std::max(std::abs(x[0]), std::abs(x[nx - 1]));

    if (xget < x[0] - zxtol || xget > x[nx - 1] + zxtol) {
      ier = 1; // Error code for out of range
      printf("cspevx:  xget=%.6f out of range %.6f to %.6f\n", xget, x[0], x[nx - 1]);
      
      return;
    } else {
      printf("cspevx:  xget=%.6f beyond range %.6f to %.6f (fixup applied)\n",
               xget, x[0], x[nx - 1]);
      zxget = (xget < x[0]) ? x[0] : x[nx - 1];
    }
  }

  int ii = static_cast<int>(nxm * (zxget - x[0]) / (x[nx - 1] - x[0]));
  i = std::min(nxm - 1, ii); // Fortran is 1-based, C++ is 0-based

  if (zxget < x[i]) {
    i = std::max(0, i - 1);
  } else if (zxget > x[i + 1]) {
    i = std::min(nxm - 1, i + 1);
  }

  dx = zxget - x[i];
}

void ibc_ck(int ibc, const std::string &slbl, const std::string &xlbl, int imin,
            int imax) {
  if (ibc < imin || ibc > imax) {
    std::ostringstream error_msg;
    error_msg << "Index out of range: " << slbl << " -- ibc" << xlbl 
              << " = " << ibc << " (valid range: " << imin << " to " << imax << ")";
    throw std::out_of_range(error_msg.str());
  }
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::mkspline(
    Rank1View<T, MemorySpace> x, const int &nx,
    Rank2View<T, MemorySpace> fspl, Rank2View<T, MemorySpace> fspl4,
    const int &ibcxmin, const double &bcxmin, const int &ibcxmax,
    const double &bcxmax, Rank1View<T, MemorySpace> wk) {
  // TODO: input check, size check and ascending

  // Copy f data to fspl4 and zero out second derivative output
  for (int i = 0; i < nx; ++i) {
    fspl4(0, i) = fspl(0, i);
    fspl(1, i) = 0.0;
  }
  // Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, nx),
  // KOKKOS_LAMBDA(const int i) {
  //     fspl4(0, i) = fspl(0, i);
  //     fspl(1, i) = 0.0;
  // });

  int inwk = nx;

  // Call traditional spline generator
  cspline(x, nx, fspl4, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);

  for (int i = 0; i < nx - 1; ++i) {
    fspl(1, i) = 2.0 * fspl4(2, i);
  }
  // Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, nx - 1),
  // KOKKOS_LAMBDA(const int i) {
  //     fspl(1, i) = 2.0 * fspl4(2, i);
  // });

  fspl(1, nx - 1) =
      2.0 * fspl4(2, nx - 2) + (x[nx - 1] - x[nx - 2]) * 6.0 * fspl4(3, nx - 2);

  fs2_ = fspl;
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::evaluate_explicit(
    const Kokkos::View<int*, MemorySpace> selector,
    Rank1View<T, MemorySpace> xvec,
    Rank2View<T, MemorySpace> fval) {
      int ivd = fval.extent(1);
      // for (int i = 0; i < xvec.extent(0); ++i){
      //   int ier = 0;
      //   auto fval_view = Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd, ivd);
      //   cspeval(xvec(i), selector, fval_view, ier);
      // }
      Kokkos::parallel_for(
          Kokkos::RangePolicy<execution_space>(0, xvec.extent(0)),
          KOKKOS_LAMBDA(const int i) {
            int ier = 0;
            auto fval_view = Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd, ivd);
            cspeval(xvec(i), selector, fval_view, ier);
          });
    }

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::evaluate_compact(
    Kokkos::View<int*, MemorySpace> selector,
    Rank1View<T, MemorySpace> xvec,
    Rank2View<T, MemorySpace> fval) {
      int ivd = fval.extent(1);
      // for (int i = 0; i < xvec.extent(0); ++i){
      //   int ier = 0;
      //   auto fval_view = Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd, ivd);
      //   evspline(xvec(i), selector, fval_view, ier);
      // }
      Kokkos::parallel_for(
          Kokkos::RangePolicy<execution_space>(0, xvec.extent(0)),
          KOKKOS_LAMBDA(const int i) {
            int ier = 0;
            auto fval_view = Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd, ivd);
            evspline(xvec(i), selector, fval_view, ier);
          });
    }


template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void CubicSplineInterpolator<T, MemorySpace>::fvspline(
    Kokkos::View<int*, MemorySpace> selector,
    Rank1View<T, MemorySpace> fval, const int &i, const double &xparam,
    const double &hx, const double &hxi) {
  const double sixth = 1.0 / 6.0;
  int iadr = 0;

  if (selector[0] <= 2) {
    if (selector[0] == 1) {
      // Function value f(x)
      ++iadr;
      // TODO: too much temp varibale created?
      double xp = xparam;
      double xpi = 1.0 - xp;
      double xp2 = xp * xp;
      double xpi2 = xpi * xpi;
      double cx = xp * (xp2 - 1.0);
      double cxi = xpi * (xpi2 - 1.0);
      double hx2 = hx * hx;

      double sum = xpi * fs2_(0, i) + xp * fs2_(0, i + 1);
      sum += sixth * hx2 * (cxi * fs2_(1, i) + cx * fs2_(1, i + 1));

      fval(iadr - 1) = sum; // zero-based indexing
    }

    if (selector[1] == 1) {
      // First derivative df/dx
      ++iadr;
      double xp = xparam;
      double xpi = 1.0 - xp;
      double xp2 = xp * xp;
      double xpi2 = xpi * xpi;

      double cxd = 3.0 * xp2 - 1.0;
      double cxdi = -3.0 * xpi2 + 1.0;

      double sum = hxi * (fs2_(0, i + 1) - fs2_(0, i));
      sum += sixth * hx * (cxdi * fs2_(1, i) + cxd * fs2_(1, i + 1));

      fval(iadr - 1) = sum;
    }

    if (selector[2] == 1) {
      // Second derivative d2f/dx2
      ++iadr;
      double xp = xparam;
      double xpi = 1.0 - xp;

      double sum = xpi * fs2_(1, i) + xp * fs2_(1, i + 1);
      fval(iadr - 1) = sum;
    }

  } else {
    // Third derivative d3f/dx3
    iadr = 1;
    fval(iadr - 1) = hxi * (fs2_(1, i + 1) - fs2_(1, i));
  }
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void CubicSplineInterpolator<T, MemorySpace>::herm1x(
    double xget, Rank1View<T, MemorySpace> x, const int &nx, int &i,
    double &xparam, double &hx, double &hxi, int &ier) {
  ier = 0;

  double zxget = xget;

  if ((xget < x[0]) || (xget > x[nx - 1])) {
    double zxtol = 4.0e-7 * std::max(std::abs(x[0]), std::abs(x[nx - 1]));
    if ((xget < x[0] - zxtol) || (xget > x[nx - 1] + zxtol)) {
      ier = 1;
      printf("herm1ev:  xget=%.6f out of range %.6f to %.6f\n", xget, x[0], x[nx - 1]);
      return;
    } else {
      if ((xget < x[0] - 0.5 * zxtol) || (xget > x[nx - 1] + 0.5 * zxtol)) {
        printf("herm1ev:  xget=%.6f beyond range %.6f to %.6f (fixup applied)\n",
               xget, x[0], x[nx - 1]);
      }
      zxget = (xget < x[0]) ? x[0] : x[nx - 1];
    }
  }

  int nxm = nx - 1;

  // TODO: potential duplicate code
  int ii = static_cast<int>(nxm * (zxget - x[0]) / (x[nx - 1] - x[0]));
  i = std::min(nxm - 1, ii);
  if (zxget < x[i]) {
    i = std::max(0, i - 1);
  } else if (zxget > x[i + 1]) {
    i = std::min(nxm - 2, i + 1);
  }

  hx = x[i + 1] - x[i];
  hxi = 1.0 / hx;
  xparam = (zxget - x[i]) * hxi;
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::evspline(
    double xget, Kokkos::View<int*, MemorySpace> ict, Rank1View<T, MemorySpace> fval,
    int &ier) {

  // Initialize output zone info
  int i = 0;
  double xparam = 0.0;
  double hx = 0.0;
  double hxi = 0.0;

  // Find the interval containing xget
  herm1x(xget, x_, nx_, i, xparam, hx, hxi, ier);
  if (ier != 0)
    return;

  // Evaluate spline at the point
  fvspline(ict, fval, i, xparam, hx, hxi);
}

template <typename T, typename MemorySpace>
class BiCubicSplineInterpolator
    : public CubicSplineInterpolator<T, MemorySpace> {
public:
  using typename SplineInterpolator<T, MemorySpace>::execution_space;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

  Rank4View<T, MemorySpace> fspl_;
  Rank3View<T, MemorySpace> f_;
  Rank1View<T, MemorySpace> x_;
  Rank1View<T, MemorySpace> y_;
  int nx_, ny_;

  BiCubicSplineInterpolator() = default;

  BiCubicSplineInterpolator(
      Rank1View<T, MemorySpace> x,           // size: inx
      int inx, Rank1View<T, MemorySpace> th, // size: inth
      int inth, Rank4View<T, MemorySpace> fspl,    // [4, 4, inx, inth]
      int ibcxmin,
      Rank1View<T, MemorySpace> bcxmin, // size: inth (used if ibcxmin = 1 or 2)
      int ibcxmax,
      Rank1View<T, MemorySpace> bcxmax, // size: inth (used if ibcxmax = 1 or 2)
      int ibcthmin,
      Rank1View<T, MemorySpace>
          bcthmin, // size: inx (used if ibcthmin = 1 or 2)
      int ibcthmax,
      Rank1View<T, MemorySpace>
          bcthmax,                  // size: inx (used if ibcthmax = 1 or 2)
      Rank1View<T, MemorySpace> wk // size: nwk
      );
    
    BiCubicSplineInterpolator(Rank1View<T, MemorySpace> x, int nx,
      Rank1View<T, MemorySpace> y, int ny,
      Rank3View<T, MemorySpace> f, int ibcxmin,
      Rank1View<T, MemorySpace> bcxmin, int ibcxmax,
      Rank1View<T, MemorySpace> bcxmax, int ibcymin,
      Rank1View<T, MemorySpace> bcymin, int ibcymax,
      Rank1View<T, MemorySpace> bcymax, Rank1View<T, MemorySpace> wk);
  
  void bcspline(
    Rank1View<T, MemorySpace> x,           // size: inx
    int inx, Rank1View<T, MemorySpace> th, // size: inth
    int inth, Rank4View<T, MemorySpace> fspl,    // [4, 4, inx, inth]
    int ibcxmin,
    Rank1View<T, MemorySpace> bcxmin, // size: inth (used if ibcxmin = 1 or 2)
    int ibcxmax,
    Rank1View<T, MemorySpace> bcxmax, // size: inth (used if ibcxmax = 1 or 2)
    int ibcthmin,
    Rank1View<T, MemorySpace>
        bcthmin, // size: inx (used if ibcthmin = 1 or 2)
    int ibcthmax,
    Rank1View<T, MemorySpace>
        bcthmax,                  // size: inx (used if ibcthmax = 1 or 2)
    Rank1View<T, MemorySpace> wk // size: nwk
    );
  
  void mkbicub(Rank1View<T, MemorySpace> x, int nx,
    Rank1View<T, MemorySpace> y, int ny,
    Rank3View<T, MemorySpace> f, int ibcxmin,
    Rank1View<T, MemorySpace> bcxmin, int ibcxmax,
    Rank1View<T, MemorySpace> bcxmax, int ibcymin,
    Rank1View<T, MemorySpace> bcymin, int ibcymax,
    Rank1View<T, MemorySpace> bcymax, Rank1View<T, MemorySpace> wk);

  void bcspeval(double xget, double yget, Kokkos::View<int*, MemorySpace> iselect,
                Rank1View<T, MemorySpace> fval,
                int &ier);

  KOKKOS_FUNCTION
  void bcspevxy(double xget, double yget,
                Rank1View<T, MemorySpace> x, int nx,
                Rank1View<T, MemorySpace> y, int ny, int &i,
                int &j, double &dx, double &dy, int &ier);

  KOKKOS_FUNCTION
  void bcspevfn(
      Kokkos::View<int*, MemorySpace> ict, // Selector array for which derivatives to compute
      Rank1View<T, MemorySpace> fval, // Output array: size [ivd, *] (flattened)
      const int &i,                   // Grid cell indices in x direction
      const int &j,                   // Grid cell indices in y direction
      const double &dx,               // x displacements within cells
      const double &dy               // y displacements within cells
  );

  KOKKOS_FUNCTION
  void herm2xy(double xget, double yget, Rank1View<T, MemorySpace> x,
               int &nx, Rank1View<T, MemorySpace> y, int &ny, int &i,
               int &j, double &xparam, double &yparam, double &hx, double &hxi,
               double &hy, double &hyi, int &ier);
  
  KOKKOS_FUNCTION
  void fvbicub(Kokkos::View<int*, MemorySpace> ict, int ivec, int ivecd,
               Rank1View<T, MemorySpace> fval, const int &i, const int &j,
               const double &xparam, const double &yparam, const double &hx,
               const double &hxi, const double &hy, const double &hyi);
  
  void evbicub(double xget, double yget,
               Kokkos::View<int*, MemorySpace> ict,
               Rank1View<T, MemorySpace> fval, // output (size depends on ict)
               int &ier);
  
  void evaluate_explicit(
    Kokkos::View<int*, MemorySpace> iselect, Rank1View<T, MemorySpace> xvec,
    Rank1View<T, MemorySpace> yvec, Rank2View<T, MemorySpace> fval);
  void evaluate_compact(
    Kokkos::View<int*, MemorySpace> iselect, Rank1View<T, MemorySpace> xvec,
    Rank1View<T, MemorySpace> yvec, Rank2View<T, MemorySpace> fval);
};

template <typename T, typename MemorySpace>
BiCubicSplineInterpolator<T, MemorySpace>::BiCubicSplineInterpolator(
    Rank1View<T, MemorySpace> x,           // size: inx
    int inx, Rank1View<T, MemorySpace> th, // size: inth
    int inth, Rank4View<T, MemorySpace> fspl,    // [4, 4, inx, inth]
    int ibcxmin,
    Rank1View<T, MemorySpace> bcxmin, // size: inth (used if ibcxmin = 1 or 2)
    int ibcxmax,
    Rank1View<T, MemorySpace> bcxmax, // size: inth (used if ibcxmax = 1 or 2)
    int ibcthmin,
    Rank1View<T, MemorySpace> bcthmin, // size: inx (used if ibcthmin = 1 or 2)
    int ibcthmax,
    Rank1View<T, MemorySpace> bcthmax, // size: inx (used if ibcthmax = 1 or 2)
    Rank1View<T, MemorySpace> wk      // size: nwk
    ) {
      x_ = x;
      nx_ = inx;
      y_ = th;
      ny_ = inth;
      bcspline(x, inx, th, inth, fspl, ibcxmin, bcxmin, ibcxmax, bcxmax, ibcthmin, bcthmin, ibcthmax, bcthmax, wk);
      fspl_ = fspl;
    }



template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::bcspline(
    Rank1View<T, MemorySpace> x,           // size: inx
    int inx, Rank1View<T, MemorySpace> th, // size: inth
    int inth, Rank4View<T, MemorySpace> fspl,    // [4, 4, inx, inth]
    int ibcxmin,
    Rank1View<T, MemorySpace> bcxmin, // size: inth (used if ibcxmin = 1 or 2)
    int ibcxmax,
    Rank1View<T, MemorySpace> bcxmax, // size: inth (used if ibcxmax = 1 or 2)
    int ibcthmin,
    Rank1View<T, MemorySpace> bcthmin, // size: inx (used if ibcthmin = 1 or 2)
    int ibcthmax,
    Rank1View<T, MemorySpace> bcthmax, // size: inx (used if ibcthmax = 1 or 2)
    Rank1View<T, MemorySpace> wk      // size: nwk
    ) {
  int iflg2 = 0;
  // std::vector<int> iselect1(10, 0); // Size 10, all initialized to 0
  // std::vector<int> iselect2(10, 0); // Size 10, all initialized to 0
  Kokkos::View<int*, MemorySpace> iselect1("iselect1", 10);
  Kokkos::deep_copy(iselect1, 0);
  Kokkos::View<int*, MemorySpace> iselect2("iselect2", 10);
  Kokkos::deep_copy(iselect2, 0);

  auto fspl_l_x = Rank2View<T, MemorySpace>(wk.data_handle() + 4 * inx * inth,
                                            4 * inx, inth);
  auto wk_l = Rank1View<T, MemorySpace>(wk.data_handle() + 2 * 4 * inx * inth,
                                        inx * inth);
  auto fspl_l_th = Rank2View<T, MemorySpace>(wk.data_handle(), 4 * inx, inth);

  if (ibcthmin != -1) {
    if (ibcthmin == 1 || ibcthmin == 2) {
      for (int ix = 0; ix < inx; ++ix) {
        if (bcthmin[ix] != 0.0) {
          iflg2 = 1;
          break;
        }
      }
    }
    if (ibcthmax == 1 || ibcthmax == 2) {
      for (int ix = 0; ix < inx; ++ix) {
        if (bcthmax[ix] != 0.0) {
          iflg2 = 1;
          break;
        }
      }
    }
  }

  int ier_tmp = 0;
  int itest = 5 * std::max(inx, inth);
  if (iflg2 == 1) {
    itest += 4 * inx * inth;
  }

  PCMS_ALWAYS_ASSERT(wk.extent(0) >= itest, MPI_COMM_WORLD,
                     "bcspline: workspace too small");
  PCMS_ALWAYS_ASSERT(inx >= 2, MPI_COMM_WORLD,
                     "bcspline: at least 2 x points required.");
  PCMS_ALWAYS_ASSERT(inth >= 2, MPI_COMM_WORLD,
                     "bcspline: need at least 2 theta points.");

  // Check boundary condition values
  ibc_ck(ibcxmin, "bcspline", "xmin", -1, 7);
  if (ibcxmin >= 0)
    ibc_ck(ibcxmax, "bcspline", "xmax", 0, 7);
  ibc_ck(ibcthmin, "bcspline", "thmin", -1, 7);
  if (ibcthmin >= 0)
    ibc_ck(ibcthmax, "bcspline", "thmax", 0, 7);

  // Check vector spacing
  CubicSplineInterpolator<T, MemorySpace>::splinck(x, 1.0e-3);
  CubicSplineInterpolator<T, MemorySpace>::splinck(th, 1.0e-3);

  double xo2 = 0.5;
  double xo6 = 1.0 / 6.0;

  // Spline in x-direction
  int inxo = 4 * (inx - 1);

  // for (int ith = 0; ith < inth; ++ith) {
  //     // Copy function into workspace
  //     for (int ix = 0; ix < inx; ++ix) {
  //         fspl_x(0, ix) = fspl(0, 0, ix, ith); // fspl(1,1,ix,ith)
  //     }

  //     // Boundary condition at xmin
  //     if (ibcxmin == 1) {
  //         fspl_x(1, 0) = bcxmin[ith];
  //     } else if (ibcxmin == 2) {
  //         fspl_x(2, 0) = bcxmin[ith];
  //     }

  //     // Boundary condition at xmax
  //     if (ibcxmax == 1) {
  //         fspl_x(1, inx - 1) = bcxmax[ith];
  //     } else if (ibcxmax == 2) {
  //         fspl_x(2, inx - 1) = bcxmax[ith];
  //     }

  //     // Call v_spline
  //     BiCubicSplineInterpolator<T, MemorySpace>::v_spline(ibcxmin, ibcxmax,
  //     inx, x, fspl_x, wk_x);

  //     // Copy coefficients out
  //     for (int ix = 0; ix < inx; ++ix) {
  //         fspl(1, 0, ix, ith) = fspl_x(1, ix);               // fspl(2,1,...)
  //         fspl(2, 0, ix, ith) = fspl_x(2, ix) * xo2;          //
  //         fspl(3,1,...) fspl(3, 0, ix, ith) = fspl_x(3, ix) * xo6; //
  //         fspl(4,1,...)
  //     }
  // }

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(inth, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &team) {
        const int ith = team.league_rank();

        auto fspl_x_view = Rank2View<T, MemorySpace>(
            fspl_l_x.data_handle() + 4 * inx * ith, 4, inx);
        auto wk_x_view =
            Rank1View<T, MemorySpace>(wk_l.data_handle() + ith * inx, inx);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, inx), [=](int ix) {
              fspl_x_view(0, ix) = fspl(0, 0, ix, ith);
            });

        if (team.team_rank() == 0) {
          if (ibcxmin == 1)
            fspl_x_view(1, 0) = bcxmin[ith];
          else if (ibcxmin == 2)
            fspl_x_view(2, 0) = bcxmin[ith];

          if (ibcxmax == 1)
            fspl_x_view(1, inx - 1) = bcxmax[ith];
          else if (ibcxmax == 2)
            fspl_x_view(2, inx - 1) = bcxmax[ith];

          BiCubicSplineInterpolator<T, MemorySpace>::v_spline(
              ibcxmin, ibcxmax, inx, x, fspl_x_view, wk_x_view);
        }

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, inx), [=](int ix) {
              fspl(1, 0, ix, ith) = fspl_x_view(1, ix);
              fspl(2, 0, ix, ith) = fspl_x_view(2, ix) * xo2;
              fspl(3, 0, ix, ith) = fspl_x_view(3, ix) * xo6;
            });
      });

  // Spline in theta direction
  int intho = 4 * (inth - 1);

  // for (int ix = 0; ix < inx; ++ix) {
  //     // Spline each x coefficient
  //     for (int ic = 0; ic < 4; ++ic) {
  //         // Copy ordinates into workspace
  //         for (int ith = 0; ith < inth; ++ith) {
  //             fspl_th(0, ith) = fspl(ic, 0, ix, ith); // fspl(ic,1,ix,ith)
  //         }

  //         // Set linear BCs initially
  //         fspl_th(1, 0) = 0.0;
  //         fspl_th(2, 0) = 0.0;
  //         fspl_th(1, inth - 1) = 0.0;
  //         fspl_th(2, inth - 1) = 0.0;

  //         // Adjust BC flags if needed
  //         int ibcthmina = ibcthmin;
  //         int ibcthmaxa = ibcthmax;
  //         if (iflg2 == 1) {
  //             if (ibcthmin == 1 || ibcthmin == 2) ibcthmina = 0;
  //             if (ibcthmax == 1 || ibcthmax == 2) ibcthmaxa = 0;
  //         }

  //         // Call v_spline
  //         BiCubicSplineInterpolator<T, MemorySpace>::v_spline(ibcthmina,
  //         ibcthmaxa, inth, th, fspl_th, wk_th);

  //         // Copy coefficients out
  //         for (int ith = 0; ith < inth; ++ith) {
  //             fspl(ic, 1, ix, ith) = fspl_th(1, ith);              //
  //             fspl(ic,2,...) fspl(ic, 2, ix, ith) = fspl_th(2, ith) * xo2; //
  //             fspl(ic,3,...) fspl(ic, 3, ix, ith) = fspl_th(3, ith) * xo6; //
  //             fspl(ic,4,...)
  //         }
  //     }
  // }
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(inx, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &team) {
        const int ix = team.league_rank();

        auto fspl_th_view = Rank2View<T, MemorySpace>(
            fspl_l_th.data_handle() + 4 * ix * inth, 4, inth);
        auto wk_th_view =
            Rank1View<T, MemorySpace>(wk_l.data_handle() + ix * inth, inth);

        for (int ic = 0; ic < 4; ++ic) {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, inth), [=](int ith) {
                fspl_th_view(0, ith) = fspl(ic, 0, ix, ith);
              });

          // Set linear BCs initially
          if (team.team_rank() == 0) {
            fspl_th_view(1, 0) = 0.0;
            fspl_th_view(2, 0) = 0.0;
            fspl_th_view(1, inth - 1) = 0.0;
            fspl_th_view(2, inth - 1) = 0.0;

            int ibcthmina = ibcthmin;
            int ibcthmaxa = ibcthmax;
            if (iflg2 == 1) {
              if (ibcthmin == 1 || ibcthmin == 2)
                ibcthmina = 0;
              if (ibcthmax == 1 || ibcthmax == 2)
                ibcthmaxa = 0;
            }

            BiCubicSplineInterpolator<T, MemorySpace>::v_spline(
                ibcthmina, ibcthmaxa, inth, th, fspl_th_view, wk_th_view);
          }

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, inth), [=](int ith) {
                fspl(ic, 1, ix, ith) = fspl_th_view(1, ith);
                fspl(ic, 2, ix, ith) = fspl_th_view(2, ith) * xo2;
                fspl(ic, 3, ix, ith) = fspl_th_view(3, ith) * xo6;
              });
        }
      });

  fspl_ = fspl;

  if (iflg2 == 1) {
    int iasc = 0;        // Workspace base for correction splines
    int iinc = 4 * inth; // Spacing between correction splines
    int iawk = iasc + 4 * inth * inx;

    double zhxn = x[inx - 1] - x[inx - 2];
    int jx = inx - 2;
    double zhth = th[inth - 1] - th[inth - 2];
    int jth = inth - 2;

    for (int ii = 0; ii < 10; ++ii) {
      iselect1[ii] = 0;
      iselect2[ii] = 0;
    }

    if (ibcthmin == 1)
      iselect1[2] = 1;
    if (ibcthmin == 2)
      iselect1[4] = 1;
    if (ibcthmax == 1)
      iselect2[2] = 1;
    if (ibcthmax == 2)
      iselect2[4] = 1;

    // for (int ix = 0; ix < inx; ++ix) {
    //     double zdiff1 = 0.0, zdiff2 = 0.0;

    //     if (ibcthmin == 1) {
    //         zcur_vec[0] = (ix < inx - 1) ? fspl(0, 1, ix, 0)
    //                                 : fspl(0, 1, jx, 0) + zhxn * (fspl(1, 1,
    //                                 jx, 0) + zhxn * (fspl(2, 1, jx, 0) + zhxn
    //                                 * fspl(3, 1, jx, 0)));
    //         zdiff1 = bcthmin[ix] - zcur_vec[0];
    //     } else if (ibcthmin == 2) {
    //         zcur_vec[0] = (ix < inx - 1) ? 2.0 * fspl(0, 2, ix, 0)
    //                                 : 2.0 * (fspl(0, 2, jx, 0) + zhxn *
    //                                 (fspl(1, 2, jx, 0) + zhxn * (fspl(2, 2,
    //                                 jx, 0) + zhxn * fspl(3, 2, jx, 0))));
    //         zdiff1 = bcthmin[ix] - zcur_vec[0];
    //     }

    //     auto zcur = Rank2View<double, MemorySpace>(zcur_vec.data(), 1,
    //     3); if (ibcthmax == 1) {
    //         if (ix < inx - 1) {
    //             zcur(0, 0) = fspl(0, 1, ix, jth) + zhth * (2.0 * fspl(0, 2,
    //             ix, jth) + zhth * 3.0 * fspl(0, 3, ix, jth));
    //         } else {
    //             // TODO: check if this is correct
    //             bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, ier); if (ier != 0)
    //             return;
    //         }
    //         zdiff2 = bcthmax[ix] - zcur(0, 0);
    //     } else if (ibcthmax == 2) {
    //         if (ix < inx - 1) {
    //             zcur(0, 0) = 2.0 * fspl(0, 2, ix, jth) + 6.0 * zhth * fspl(0,
    //             3, ix, jth);
    //         } else {
    //             // TODO: check if this is correct
    //             bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, ier);
    //             if (ier != 0)
    //             return;
    //         }
    //         zdiff2 = bcthmax[ix] - zcur(0, 0);
    //     }

    //     int iadr = iasc + ix * iinc;
    //     for (int ith = 0; ith < inth; ++ith)
    //         fspl_l_th(ix * 4, ith) = 0.0;

    //     fspl_l_th(1 + ix * 4, 0) = 0.0;
    //     fspl_l_th(2 + ix * 4, 0) = 0.0;
    //     fspl_l_th(1 + ix * 4, inth - 1) = 0.0;
    //     fspl_l_th(2 + ix * 4, inth - 1) = 0.0;

    //     if (ibcthmin == 1) fspl_l_th(1 + ix * 4, 0) = zdiff1;
    //     else if (ibcthmin == 2) fspl_l_th(2 + ix * 4, 0) = zdiff1;

    //     if (ibcthmax == 1) fspl_l_th(1 + ix * 4, inth - 1) = zdiff2;
    //     else if (ibcthmax == 2) fspl_l_th(2 + ix * 4, inth - 1) = zdiff2;

    //     auto fspl_s_th = Rank2View<double,
    //     MemorySpace>(fspl_l_th.data_handle() + iadr, 4, inth);
    //     BiCubicSplineInterpolator<T, MemorySpace>::v_spline(ibcthmin,
    //     ibcthmax, inth, th, fspl_s_th, wk_th);

    // }

    auto zcur_shared = Rank1View<T, MemorySpace>(wk.data_handle() + 9 * inx * inth,
                                        inx * 3);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, inx),
        KOKKOS_LAMBDA(const int ix) {
          int ier = 0;
          auto zcur = Rank1View<T, MemorySpace>(zcur_shared.data_handle() + ix * 3,
                                        3);
          double zdiff1 = 0.0, zdiff2 = 0.0;
          auto wk_th = Rank1View<T, MemorySpace>(wk_l.data_handle(), inth);

          if (ibcthmin == 1) {
            zcur[0] = (ix < inx - 1)
                              ? fspl(0, 1, ix, 0)
                              : fspl(0, 1, jx, 0) +
                                    zhxn * (fspl(1, 1, jx, 0) +
                                            zhxn * (fspl(2, 1, jx, 0) +
                                                    zhxn * fspl(3, 1, jx, 0)));
            zdiff1 = bcthmin[ix] - zcur[0];
          } else if (ibcthmin == 2) {
            zcur[0] =
                (ix < inx - 1)
                    ? 2.0 * fspl(0, 2, ix, 0)
                    : 2.0 * (fspl(0, 2, jx, 0) +
                             zhxn * (fspl(1, 2, jx, 0) +
                                     zhxn * (fspl(2, 2, jx, 0) +
                                             zhxn * fspl(3, 2, jx, 0))));
            zdiff1 = bcthmin[ix] - zcur[0];
          }

          // auto zcur = Rank1View<T, MemorySpace>(zcur_vec.data(), 3);
          if (ibcthmax == 1) {
            if (ix < inx - 1) {
              zcur(0) = fspl(0, 1, ix, jth) +
                        zhth * (2.0 * fspl(0, 2, ix, jth) +
                                zhth * 3.0 * fspl(0, 3, ix, jth));
            } else {
              // TODO: check if this is correct
              bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, ier);
              if (ier != 0)
                return;
            }
            zdiff2 = bcthmax[ix] - zcur(0);
          } else if (ibcthmax == 2) {
            if (ix < inx - 1) {
              zcur(0) =
                  2.0 * fspl(0, 2, ix, jth) + 6.0 * zhth * fspl(0, 3, ix, jth);
            } else {
              // TODO: check if this is correct
              bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, ier);
              if (ier != 0)
                return;
            }
            zdiff2 = bcthmax[ix] - zcur(0);
          }

          int iadr = iasc + ix * iinc;
          for (int ith = 0; ith < inth; ++ith)
            fspl_l_th(ix * 4, ith) = 0.0;

          fspl_l_th(1 + ix * 4, 0) = 0.0;
          fspl_l_th(2 + ix * 4, 0) = 0.0;
          fspl_l_th(1 + ix * 4, inth - 1) = 0.0;
          fspl_l_th(2 + ix * 4, inth - 1) = 0.0;

          if (ibcthmin == 1)
            fspl_l_th(1 + ix * 4, 0) = zdiff1;
          else if (ibcthmin == 2)
            fspl_l_th(2 + ix * 4, 0) = zdiff1;

          if (ibcthmax == 1)
            fspl_l_th(1 + ix * 4, inth - 1) = zdiff2;
          else if (ibcthmax == 2)
            fspl_l_th(2 + ix * 4, inth - 1) = zdiff2;

          auto fspl_s_th = Rank2View<T, MemorySpace>(
              fspl_l_th.data_handle() + iadr, 4, inth);
          BiCubicSplineInterpolator<T, MemorySpace>::v_spline(
              ibcthmin, ibcthmax, inth, th, fspl_s_th, wk_th);
        });

    // for (int ix = 0; ix < inx; ++ix) {
    //     int iadr = iasc + ix * iinc;
    //     for (int ith = 0; ith < inth - 1; ++ith) {
    //         fspl_l_th(2 + ix * 4, ith) *= xo2;
    //         fspl_l_th(3 + ix * 4, ith) *= xo6;
    //         if (ix < inx - 1) {
    //             fspl(0, 1, ix, ith) += fspl_l_th(1 + ix * 4, ith);
    //             fspl(0, 2, ix, ith) += fspl_l_th(2 + ix * 4, ith);
    //             fspl(0, 3, ix, ith) += fspl_l_th(3 + ix * 4, ith);
    //         }
    //     }
    // }

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {inx, inth - 1}),
        KOKKOS_LAMBDA(const int ix, const int ith) {
          // Multiply local coefficients
          fspl_l_th(2 + ix * 4, ith) *= xo2;
          fspl_l_th(3 + ix * 4, ith) *= xo6;

          // Conditional accumulation into global fspl
          if (ix < inx - 1) {
            fspl(0, 1, ix, ith) += fspl_l_th(1 + ix * 4, ith);
            fspl(0, 2, ix, ith) += fspl_l_th(2 + ix * 4, ith);
            fspl(0, 3, ix, ith) += fspl_l_th(3 + ix * 4, ith);
          }
        });

    int ia5w = iawk + 4 * inx;

    // for (int ith = 0; ith < inth - 1; ++ith) {
    //     for (int ic = 1; ic < 4; ++ic) {
    //         for (int ix = 0; ix < inx; ++ix) {
    //             int iaspl = iasc + iinc * ix;
    //             fspl_x(0, ix) = fspl_l_th(ic + ix * 4, ith);
    //         }
    //         fspl_x(1, 0) = 0.0;
    //         fspl_x(2, 0) = 0.0;
    //         fspl_x(1, inx - 1) = 0.0;
    //         fspl_x(2, inx - 1) = 0.0;

    //         BiCubicSplineInterpolator<T, MemorySpace>::v_spline(ibcxmin,
    //         ibcxmax, inx, x, fspl_x, wk_x);

    //         for (int ix = 0; ix < inx - 1; ++ix) {
    //             fspl(1, ic, ix, ith) += fspl_x(1, ix);
    //             fspl(2, ic, ix, ith) += fspl_x(2, ix) * xo2;
    //             fspl(3, ic, ix, ith) += fspl_x(3, ix) * xo6;
    //         }
    //     }
    // }
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<execution_space>(inth - 1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type &team) {
          const int ith = team.league_rank();

          auto fspl_x_view = Rank2View<T, MemorySpace>(
              fspl_l_x.data_handle() + 4 * inx * ith, 4, inx);
          auto wk_x_view =
              Rank1View<T, MemorySpace>(wk_l.data_handle() + ith * inx, inx);

          for (int ic = 1; ic < 4; ++ic) {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, inx), [=](int ix) {
                  fspl_x_view(0, ix) = fspl_l_th(ic + ix * 4, ith);
                });

            if (team.team_rank() == 0) {
              fspl_x_view(1, 0) = 0.0;
              fspl_x_view(2, 0) = 0.0;
              fspl_x_view(1, inx - 1) = 0.0;
              fspl_x_view(2, inx - 1) = 0.0;

              BiCubicSplineInterpolator<T, MemorySpace>::v_spline(
                  ibcxmin, ibcxmax, inx, x, fspl_x_view, wk_x_view);
            }

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, inx - 1), [=](int ix) {
                  fspl(1, ic, ix, ith) += fspl_x_view(1, ix);
                  fspl(2, ic, ix, ith) += fspl_x_view(2, ix) * xo2;
                  fspl(3, ic, ix, ith) += fspl_x_view(3, ix) * xo6;
                });
          }
        });
  }
}

template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::evaluate_explicit(
  Kokkos::View<int*, MemorySpace> iselect, Rank1View<T, MemorySpace> xvec,
  Rank1View<T, MemorySpace> yvec, Rank2View<T, MemorySpace> fval) {
  PCMS_ALWAYS_ASSERT(
      xvec.extent(0) == yvec.extent(0) && xvec.extent(0) == fval.extent(0),
      MPI_COMM_WORLD,
      "BiCubicSplineInterpolator: Input vectors must have the same length.\n");
  
  int ivd = fval.extent(1);
  // for (size_t i = 0; i < xvec.extent(0); ++i) {
  //   int ier = 0;
  //   auto fval_view = Rank1View<T, MemorySpace>(
  //       fval.data_handle() + i * ivd, ivd);
  //   bcspeval(xvec(i), yvec(i), iselect, fval_view, ier);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(0, xvec.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        int ier = 0;
        auto fval_view = Rank1View<T, MemorySpace>(
            fval.data_handle() + i * ivd, ivd);
        bcspeval(xvec(i), yvec(i), iselect, fval_view, ier);
      });
}

template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::evaluate_compact(
  Kokkos::View<int*, MemorySpace> iselect, Rank1View<T, MemorySpace> xvec,
  Rank1View<T, MemorySpace> yvec, Rank2View<T, MemorySpace> fval) {
  
  PCMS_ALWAYS_ASSERT(
      xvec.extent(0) == yvec.extent(0) && xvec.extent(0) == fval.extent(0),
      MPI_COMM_WORLD,
      "BiCubicSplineInterpolator: Input vectors must have the same length.\n");
  int ivd = fval.extent(1);
  // for (size_t i = 0; i < xvec.extent(0); ++i) {
  //   int ier = 0;
  //   auto fval_view = Rank1View<T, MemorySpace>(
  //       fval.data_handle() + i * ivd, ivd);
  //   evbicub(
  //       xvec(i), yvec(i), iselect, fval_view, ier);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(0, xvec.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        int ier = 0;
        auto fval_view = Rank1View<T, MemorySpace>(
            fval.data_handle() + i * ivd, ivd);
        evbicub(xvec(i), yvec(i), iselect, fval_view, ier);
      });
}


template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::bcspeval(
    double xget, double yget, Kokkos::View<int*, MemorySpace> iselect,
    Rank1View<T, MemorySpace> fval,
    int &ier) {
  int i = 0;
  int j = 0;
  double dx = 0.0;
  double dy = 0.0;

  // Range finding
  bcspevxy(xget, yget, x_, nx_, y_, ny_, i, j, dx, dy, ier);
  if (ier != 0)
    return;

  // Evaluate spline function
  bcspevfn(iselect, fval, i, j, dx, dy);
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void BiCubicSplineInterpolator<T, MemorySpace>::bcspevxy(
    double xget, double yget, Rank1View<T, MemorySpace> x,
    int nx, Rank1View<T, MemorySpace> y, int ny, int &i, int &j,
    double &dx, double &dy, int &ier) {
  int nxm = nx - 1;
  int nym = ny - 1;
  int ii, jj;

  ier = 0;
  double zxget = xget;
  double zyget = yget;

  if ((xget < x[0]) || (xget > x[nx - 1])) {
    double zxtol = 4.0e-7 * std::max(std::abs(x[0]), std::abs(x[nx - 1]));
    if ((xget < x[0] - zxtol) || (xget > x[nx - 1] + zxtol)) {
      ier = 1;
      printf("bcspeval: xget=%.6f out of range %.6f to %.6f\n",
             xget, x[0], x[nx - 1]);
    } else {
      if ((xget < x[0] - 0.5 * zxtol) || (xget > x[nx - 1] + 0.5 * zxtol)) {
        printf("bcspeval: xget=%.6f beyond range %.6f to %.6f (fixup applied)\n",
               xget, x[0], x[nx - 1]);
      }
      zxget = (xget < x[0]) ? x[0] : x[nx - 1];
    }
  }

  if ((yget < y[0]) || (yget > y[ny - 1])) {
    double zytol = 4.0e-7 * std::max(std::abs(y[0]), std::abs(y[ny - 1]));
    if ((yget < y[0] - zytol) || (yget > y[ny - 1] + zytol)) {
      ier = 1;
      printf("bcspeval: yget=%.6f out of range %.6f to %.6f\n",
             yget, y[0], y[nx - 1]);
    } else {
      if ((yget < y[0] - 0.5 * zytol) || (yget > y[ny - 1] + 0.5 * zytol)) {
        printf("bcspeval: yget=%.6f beyond range %.6f to %.6f (fixup applied)\n",
               yget, y[0], y[nx - 1]);
      }
      zyget = (yget < y[0]) ? y[0] : y[ny - 1];
    }
  }

  if (ier != 0)
    return;

  ii = static_cast<int>(nxm * (zxget - x[0]) / (x[nx - 1] - x[0]));
  i = std::min(nxm - 2, ii);
  if (zxget < x[i]) {
    i--;
  } else if (zxget > x[i + 1]) {
    i++;
  }

  jj = static_cast<int>(nym * (zyget - y[0]) / (y[ny - 1] - y[0]));
  j = std::min(nym - 2, jj);
  if (zyget < y[j]) {
    j--;
  } else if (zyget > y[j + 1]) {
    j++;
  }

  dx = zxget - x[i];
  dy = zyget - y[j];
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void BiCubicSplineInterpolator<T, MemorySpace>::bcspevfn(
    Kokkos::View<int*, MemorySpace>
        ict, // Selector array for which derivatives to compute
    Rank1View<T, MemorySpace> fval, // Output array: size [ivd, *] (flattened)
    const int &i,                   // Grid cell indices in x direction
    const int &j,                   // Grid cell indices in y direction
    const double &dx,               // x displacements within cells
    const double &dy               // y displacements within cells
) {
  int iaval = 0; // Index for fval
  if (ict[0] <= 2) {
    if ((ict[0] > 0) || (ict[0] == -1)) {
      // Evaluate f
      iaval++;
      fval(iaval - 1) =
          fspl_(0, 0, i, j) +
          dy * (fspl_(0, 1, i, j) +
                dy * (fspl_(0, 2, i, j) + dy * fspl_(0, 3, i, j))) +
          dx * (fspl_(1, 0, i, j) +
                dy * (fspl_(1, 1, i, j) +
                      dy * (fspl_(1, 2, i, j) + dy * fspl_(1, 3, i, j))) +
                dx * (fspl_(2, 0, i, j) +
                      dy * (fspl_(2, 1, i, j) +
                            dy * (fspl_(2, 2, i, j) + dy * fspl_(2, 3, i, j))) +
                      dx * (fspl_(3, 0, i, j) +
                            dy * (fspl_(3, 1, i, j) +
                                  dy * (fspl_(3, 2, i, j) +
                                        dy * fspl_(3, 3, i, j))))));
    }

    if ((ict[1] > 0) && (ict[0] != -1)) {
      // Evaluate df/dx
      iaval++;
      fval(iaval - 1) =
          fspl_(1, 0, i, j) +
          dy * (fspl_(1, 1, i, j) +
                dy * (fspl_(1, 2, i, j) + dy * fspl_(1, 3, i, j))) +
          2.0 * dx *
              (fspl_(2, 0, i, j) +
               dy * (fspl_(2, 1, i, j) +
                     dy * (fspl_(2, 2, i, j) + dy * fspl_(2, 3, i, j))) +
               1.5 * dx *
                   (fspl_(3, 0, i, j) +
                    dy * (fspl_(3, 1, i, j) +
                          dy * (fspl_(3, 2, i, j) + dy * fspl_(3, 3, i, j)))));
    }

    if ((ict[2] > 0) && (ict[0] != -1)) {
      // Evaluate df/dy
      iaval++;
      fval(iaval - 1) =
          fspl_(0, 1, i, j) +
          dy * (2.0 * fspl_(0, 2, i, j) + dy * 3.0 * fspl_(0, 3, i, j)) +
          dx * (fspl_(1, 1, i, j) +
                dy * (2.0 * fspl_(1, 2, i, j) + dy * 3.0 * fspl_(1, 3, i, j)) +
                dx * (fspl_(2, 1, i, j) +
                      dy * (2.0 * fspl_(2, 2, i, j) +
                            dy * 3.0 * fspl_(2, 3, i, j)) +
                      dx * (fspl_(3, 1, i, j) +
                            dy * (2.0 * fspl_(3, 2, i, j) +
                                  dy * 3.0 * fspl_(3, 3, i, j)))));
    }

    if ((ict[3] > 0) || (ict[0] == -1)) {
      // Evaluate d2f/dx2
      iaval++;
      fval(iaval - 1) =
          2.0 * (fspl_(2, 0, i, j) +
                 dy * (fspl_(2, 1, i, j) +
                       dy * (fspl_(2, 2, i, j) + dy * fspl_(2, 3, i, j)))) +
          6.0 * dx *
              (fspl_(3, 0, i, j) +
               dy * (fspl_(3, 1, i, j) +
                     dy * (fspl_(3, 2, i, j) + dy * fspl_(3, 3, i, j))));
    }

    if ((ict[4] > 0) || (ict[0] == -1)) {
      // Evaluate d2f/dy2
      iaval++;
      fval(iaval - 1) =
          2.0 * fspl_(0, 2, i, j) + 6.0 * dy * fspl_(0, 3, i, j) +
          dx * (2.0 * fspl_(1, 2, i, j) + 6.0 * dy * fspl_(1, 3, i, j) +
                dx * (2.0 * fspl_(2, 2, i, j) + 6.0 * dy * fspl_(2, 3, i, j) +
                      dx * (2.0 * fspl_(3, 2, i, j) +
                            6.0 * dy * fspl_(3, 3, i, j))));
    }

    if ((ict[5] > 0) && (ict[0] != -1)) {
      // Evaluate d2f/dxdy
      iaval++;
      fval(iaval - 1) =
          fspl_(1, 1, i, j) +
          dy * (2.0 * fspl_(1, 2, i, j) + dy * 3.0 * fspl_(1, 3, i, j)) +
          2.0 * dx *
              (fspl_(2, 1, i, j) +
               dy * (2.0 * fspl_(2, 2, i, j) + dy * 3.0 * fspl_(2, 3, i, j)) +
               1.5 * dx *
                   (fspl_(3, 1, i, j) + dy * (2.0 * fspl_(3, 2, i, j) +
                                              dy * 3.0 * fspl_(3, 3, i, j))));
    }

    if (ict[0] == -1) {
      // Evaluate d4f/dx2dy2
      iaval++;
      fval(iaval - 1) =
          4.0 * fspl_(2, 2, i, j) + 12.0 * dy * fspl_(2, 3, i, j) +
          dx * (12.0 * fspl_(3, 2, i, j) + 36.0 * dy * fspl_(3, 3, i, j));
    }
  } else if (ict[0] == 3) {
    if (ict[1] == 1) {
      // d³f/dx³ (not continuous)
      iaval++;
      fval(iaval - 1) =
          6.0 * (fspl_(3, 0, i, j) +
                 dy * (fspl_(3, 1, i, j) +
                       dy * (fspl_(3, 2, i, j) + dy * fspl_(3, 3, i, j))));
    }

    if (ict[2] == 1) {
      // d³f/dx²dy
      iaval++;
      fval(iaval - 1) =
          2.0 * (fspl_(2, 1, i, j) + dy * (2.0 * fspl_(2, 2, i, j) +
                                           dy * 3.0 * fspl_(2, 3, i, j))) +
          6.0 * dx *
              (fspl_(3, 1, i, j) +
               dy * (2.0 * fspl_(3, 2, i, j) + dy * 3.0 * fspl_(3, 3, i, j)));
    }

    if (ict[3] == 1) {
      // d³f/dxdy²
      iaval++;
      fval(iaval - 1) =
          2.0 * fspl_(1, 2, i, j) + 6.0 * dy * fspl_(1, 3, i, j) +
          2.0 * dx *
              (2.0 * fspl_(2, 2, i, j) + 6.0 * dy * fspl_(2, 3, i, j) +
               1.5 * dx *
                   (2.0 * fspl_(3, 2, i, j) + 6.0 * dy * fspl_(3, 3, i, j)));
    }

    if (ict[4] == 1) {
      // d³f/dy³ (not continuous)
      iaval++;
      fval(iaval - 1) =
          6.0 * (fspl_(0, 3, i, j) +
                 dx * (fspl_(1, 3, i, j) +
                       dx * (fspl_(2, 3, i, j) + dx * fspl_(3, 3, i, j))));
    }

  } else if (ict[0] == 4) {
    if (ict[1] == 1) {
      // d⁴f/dx³dy
      iaval++;
      fval(iaval - 1) =
          6.0 * (fspl_(3, 1, i, j) +
                 dy * 2.0 * (fspl_(3, 2, i, j) + dy * 1.5 * fspl_(3, 3, i, j)));
    }

    if (ict[2] == 1) {
      // d⁴f/dx²dy²
      iaval++;
      fval(iaval - 1) =
          4.0 * fspl_(2, 2, i, j) + 12.0 * dy * fspl_(2, 3, i, j) +
          dx * (12.0 * fspl_(3, 2, i, j) + 36.0 * dy * fspl_(3, 3, i, j));
    }

    if (ict[3] == 1) {
      // d⁴f/dxdy³ (not continuous)
      iaval++;
      fval(iaval - 1) =
          6.0 * (fspl_(1, 3, i, j) +
                 2.0 * dx * (fspl_(2, 3, i, j) + 1.5 * dx * fspl_(3, 3, i, j)));
    }

  } else if (ict[0] == 5) {
    if (ict[1] == 1) {
      // d⁵f/dx³dy² (not continuous)
      iaval++;
      fval(iaval - 1) =
          12.0 * (fspl_(3, 2, i, j) + dy * 3.0 * fspl_(3, 3, i, j));
    }

    if (ict[2] == 1) {
      // d⁵f/dx²dy³ (not continuous)
      iaval++;
      fval(iaval - 1) =
          12.0 * (fspl_(2, 3, i, j) + dx * 3.0 * fspl_(3, 3, i, j));
    }

  } else if (ict[0] == 6) {
    // d⁶f/dx³dy³ (not continuous)
    iaval++;
    fval(iaval - 1) = 36.0 * fspl_(3, 3, i, j);
  }
} // end bcspevfn


template <typename T, typename MemorySpace>
BiCubicSplineInterpolator<T, MemorySpace>::BiCubicSplineInterpolator(
    Rank1View<T, MemorySpace> x, int nx,
    Rank1View<T, MemorySpace> y, int ny, Rank3View<T, MemorySpace> f,
    int ibcxmin, Rank1View<T, MemorySpace> bcxmin, int ibcxmax,
    Rank1View<T, MemorySpace> bcxmax, int ibcymin,
    Rank1View<T, MemorySpace> bcymin, int ibcymax,
    Rank1View<T, MemorySpace> bcymax, Rank1View<T, MemorySpace> wk) {
      x_ = x; // Store the x-coordinates
      y_ = y; // Store the y-coordinates
      nx_ = nx; // Store the number of x-coordinates
      ny_ = ny; // Store the number of y-coordinates
      mkbicub(x, nx, y, ny, f, ibcxmin, bcxmin, ibcxmax, bcxmax, ibcymin, bcymin, ibcymax, bcymax, wk);
      f_ = f; // Store the spline coefficients
    }


template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::mkbicub(
    Rank1View<T, MemorySpace> x, int nx,
    Rank1View<T, MemorySpace> y, int ny, Rank3View<T, MemorySpace> f,
    int ibcxmin, Rank1View<T, MemorySpace> bcxmin, int ibcxmax,
    Rank1View<T, MemorySpace> bcxmax, int ibcymin,
    Rank1View<T, MemorySpace> bcymin, int ibcymax,
    Rank1View<T, MemorySpace> bcymax, Rank1View<T, MemorySpace> wk) {
  int iflg2 = 0;

  // Check if inhomogeneous y-boundary conditions exist
  if (ibcymin != -1) {
    if ((ibcymin == 1 || ibcymin == 2)) {
      for (int ix = 0; ix < nx; ++ix) {
        if (bcymin[ix] != 0.0)
          iflg2 = 1;
      }
    }
    if ((ibcymax == 1 || ibcymax == 2)) {
      for (int ix = 0; ix < nx; ++ix) {
        if (bcymax[ix] != 0.0)
          iflg2 = 1;
      }
    }
  }

  // Check bc validity (implement ibc_ck separately)
  ibc_ck(ibcxmin, "bcspline", "xmin", -1, 7);
  if (ibcxmin >= 0)
    ibc_ck(ibcxmax, "bcspline", "xmax", 0, 7);
  ibc_ck(ibcymin, "bcspline", "ymin", -1, 7);
  if (ibcymin >= 0)
    ibc_ck(ibcymax, "bcspline", "ymax", 0, 7);

  // Check if x and y are strictly ascending (implement splinck separately)
  CubicSplineInterpolator<T, MemorySpace>::splinck(x, 1.0e-3);
  CubicSplineInterpolator<T, MemorySpace>::splinck(y, 1.0e-3);

  auto fspl_l_x = Rank2View<T, MemorySpace>(wk.data_handle(), 2 * ny, nx);
  auto wk_l = Rank1View<T, MemorySpace>(wk.data_handle() + 2 * ny * nx, nx * ny);
  auto fwk4_l_x = Rank2View<T, MemorySpace>(
      wk.data_handle() + 2 * ny * nx + nx * ny, 4 * ny, nx);
  // // Compute fxx
  // for (int iy = 0; iy < ny; ++iy) {
  //     for (int ix = 0; ix < nx; ++ix)
  //         fwk_x(0, ix) = f(0, ix, iy);

  //     if (ibcxmin == 1 || ibcxmin == 2) zbcmin = bcxmin[iy];
  //     if (ibcxmax == 1 || ibcxmax == 2) zbcmax = bcxmax[iy];

  //     CubicSplineInterpolator<T, MemorySpace>::mkspline(x, nx, fwk_x, fwk4_x,
  //     ibcxmin, zbcmin, ibcxmax, zbcmax, wk_x); if (ier != 0) return;

  //     for (int ix = 0; ix < nx; ++ix)
  //         f(1, ix, iy) = fwk_x(1, ix);
  // }
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(ny, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &team) {
        const int iy = team.league_rank();
        double zbcmin = 0.0, zbcmax = 0.0;

        auto fwk_x_view = Rank2View<T, MemorySpace>(
            fspl_l_x.data_handle() + 2 * nx * iy, 2, nx);
        auto wk_x_view =
            Rank1View<T, MemorySpace>(wk_l.data_handle() + nx * iy, nx);
        auto fwk4_x_view = Rank2View<T, MemorySpace>(
            fwk4_l_x.data_handle() + 4 * nx * iy, 4, nx);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nx),
            [=](int ix) { fwk_x_view(0, ix) = f(0, ix, iy); });

        if (team.team_rank() == 0) {
          if (ibcxmin == 1 || ibcxmin == 2)
            zbcmin = bcxmin[iy];
          if (ibcxmax == 1 || ibcxmax == 2)
            zbcmax = bcxmax[iy];

          CubicSplineInterpolator<T, MemorySpace>::mkspline(
              x, nx, fwk_x_view, fwk4_x_view, ibcxmin, zbcmin, ibcxmax, zbcmax,
              wk_x_view);
        }

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nx),
            [=](int ix) { f(1, ix, iy) = fwk_x_view(1, ix); });
      });

  // double zbcmin = 0.0, zbcmax = 0.0;
  // int ibcmin = ibcymin, ibcmax = ibcymax;
  // // Compute fyy
  // for (int ix = 0; ix < nx; ++ix) {
  //     for (int iy = 0; iy < ny; ++iy)
  //         fwk_y(0, iy) = f(0, ix, iy);

  //     if (iflg2 == 1) {
  //         if (ibcymin == 1 || ibcymin == 2) ibcmin = 0;
  //         if (ibcymax == 1 || ibcymax == 2) ibcmax = 0;
  //     }

  //     CubicSplineInterpolator<T, MemorySpace>::mkspline(y, ny, fwk_y, fwk4_y,
  //     ibcmin, 0.0, ibcmax, 0.0, wk_y); if (ier != 0) return;

  //     for (int iy = 0; iy < ny; ++iy)
  //         f(2, ix, iy) = fwk_y(1, iy);
  // }
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(nx, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &team) {
        const int ix = team.league_rank();
        double zbcmin = 0.0, zbcmax = 0.0;
        int ibcmin = ibcymin, ibcmax = ibcymax;

        auto fwk_y_view = Rank2View<T, MemorySpace>(
            fspl_l_x.data_handle() + 2 * ny * ix, 2, ny);
        auto wk_y_view =
            Rank1View<T, MemorySpace>(wk_l.data_handle() + ny * ix, ny);
        auto fwk4_y_view = Rank2View<T, MemorySpace>(
            fwk4_l_x.data_handle() + 4 * ny * ix, 4, ny);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, ny),
            [=](int iy) { fwk_y_view(0, iy) = f(0, ix, iy); });

        if (team.team_rank() == 0) {
          if (iflg2 == 1) {
            if (ibcymin == 1 || ibcymin == 2)
              ibcmin = 0;
            if (ibcymax == 1 || ibcymax == 2)
              ibcmax = 0;
          }

          CubicSplineInterpolator<T, MemorySpace>::mkspline(
              y, ny, fwk_y_view, fwk4_y_view, ibcmin, 0.0, ibcmax, 0.0,
              wk_y_view);
        }
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, ny),
            [=](int iy) { f(2, ix, iy) = fwk_y_view(1, iy); });
      });

  // int zbcmin = 0.0;
  // int zbcmax = 0.0;
  // double ibcmin = ibcymin;
  // double ibcmax = ibcymax;
  // // Compute fxxyy
  // for (int ix = 0; ix < nx; ++ix) {
  //     for (int iy = 0; iy < ny; ++iy)
  //         fwk_y(0, iy) = f(1, ix, iy);

  //     if (iflg2 == 1) {
  //         if (ibcymin == 1 || ibcymin == 2) ibcmin = 0;
  //         if (ibcymax == 1 || ibcymax == 2) ibcmax = 0;
  //     }

  //     CubicSplineInterpolator<T, MemorySpace>::mkspline(y, ny, fwk_y, fwk4_y,
  //     ibcmin, 0.0, ibcmax, 0.0, wk_y); if (ier != 0) return;

  //     for (int iy = 0; iy < ny; ++iy)
  //         f(3, ix, iy) = fwk_y(1, iy);
  // }

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(nx, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &team) {
        const int ix = team.league_rank();
        double zbcmin = 0.0, zbcmax = 0.0;
        int ibcmin = ibcymin, ibcmax = ibcymax;

        auto fwk_y_view = Rank2View<T, MemorySpace>(
            fspl_l_x.data_handle() + 2 * ny * ix, 2, ny);
        auto wk_y_view =
            Rank1View<T, MemorySpace>(wk_l.data_handle() + ny * ix, ny);
        auto fwk4_y_view = Rank2View<T, MemorySpace>(
            fwk4_l_x.data_handle() + 4 * ny * ix, 4, ny);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, ny),
            [=](int iy) { fwk_y_view(0, iy) = f(1, ix, iy); });

        if (team.team_rank() == 0) {
          if (iflg2 == 1) {
            if (ibcymin == 1 || ibcymin == 2)
              ibcmin = 0;
            if (ibcymax == 1 || ibcymax == 2)
              ibcmax = 0;
          }

          CubicSplineInterpolator<T, MemorySpace>::mkspline(
              y, ny, fwk_y_view, fwk4_y_view, ibcmin, 0.0, ibcmax, 0.0,
              wk_y_view);
        }
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, ny),
            [=](int iy) { f(3, ix, iy) = fwk_y_view(1, iy); });
      });

  int zbcmin = 0.0;
  int zbcmax = 0.0;

  // Correct for inhomogeneous boundary conditions if needed
  if (iflg2 == 1) {
    // std::vector<T> fcorr_arr(2 * nx * ny);
    // auto fcorr = Rank3View<T, MemorySpace>(fcorr_arr.data(), 2, nx, ny);
    auto fcorr = Rank3View<T, MemorySpace>(
      wk.data_handle() + 7 * ny * nx, 2, nx, ny);
  

    // T zdiff1 = 0.0, zdiff2 = 0.0;
    // for (int ix = 0; ix < nx; ++ix) {
    //     zdiff1 = (ibcymin == 1) ? (bcymin[ix] - ((f(0, ix, 1) - f(0, ix, 0))
    //     / (y[1] - y[0]) + (y[1] - y[0]) * (-2.0 * f(2, ix, 0) - f(2, ix, 1))
    //     / 6.0)) :
    //                 ((ibcymin == 2) ? bcymin[ix] - f(2, ix, 0) : 0.0);

    //     zdiff2 = (ibcymax == 1) ? (bcymax[ix] - ((f(0, ix, ny-1) - f(0, ix,
    //     ny-2)) / (y[ny-1] - y[ny-2]) + (y[ny-1] - y[ny-2]) * (2.0 * f(2, ix,
    //     ny-1) + f(2, ix, ny-2)) / 6.0)) :
    //                 ((ibcymax == 2) ? bcymax[ix] - f(2, ix, ny-1) : 0.0);

    //     for (int iy = 0; iy < ny; ++iy) {
    //         fwk_y(0, iy) = 0.0;
    //     }
    //     CubicSplineInterpolator<T, MemorySpace>::mkspline(y, ny, fwk_y,
    //     fwk4_y, ibcymin, zdiff1, ibcymax, zdiff2, wk_y); if (ier != 0)
    //     return; for (int iy = 0; iy < ny; ++iy)
    //         fcorr(0, ix, iy) = fwk_y(1, iy);
    // }
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<execution_space>(nx, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type &team) {
          const int ix = team.league_rank();
          double zdiff1 = 0.0, zdiff2 = 0.0;

          auto fwk_y_view = Rank2View<T, MemorySpace>(
              fspl_l_x.data_handle() + 2 * ny * ix, 2, ny);
          auto wk_y_view =
              Rank1View<T, MemorySpace>(wk_l.data_handle() + ny * ix, ny);
          auto fwk4_y_view = Rank2View<T, MemorySpace>(
              fwk4_l_x.data_handle() + 4 * ny * ix, 4, ny);

          if (team.team_rank() == 0) {
            if (ibcymin == 1) {
              zdiff1 =
                  bcymin[ix] -
                  ((f(0, ix, 1) - f(0, ix, 0)) / (y[1] - y[0]) +
                   (y[1] - y[0]) * (-2.0 * f(2, ix, 0) - f(2, ix, 1)) / 6.0);
            } else if (ibcymin == 2) {
              zdiff1 = bcymin[ix] - f(2, ix, 0);
            } else {
              zdiff1 = 0.0;
            }

            if (ibcymax == 1) {
              zdiff2 = bcymax[ix] -
                       ((f(0, ix, ny - 1) - f(0, ix, ny - 2)) /
                            (y[ny - 1] - y[ny - 2]) +
                        (y[ny - 1] - y[ny - 2]) *
                            (2.0 * f(2, ix, ny - 1) + f(2, ix, ny - 2)) / 6.0);
            } else if (ibcymax == 2) {
              zdiff2 = bcymax[ix] - f(2, ix, ny - 1);
            } else {
              zdiff2 = 0.0;
            }
          }

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, ny),
              [=](int iy) { fwk_y_view(0, iy) = 0.0; });
          if (team.team_rank() == 0) {
            CubicSplineInterpolator<T, MemorySpace>::mkspline(
                y, ny, fwk_y_view, fwk4_y_view, ibcymin, zdiff1, ibcymax,
                zdiff2, wk_y_view);
          }
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, ny),
              [=](int iy) { fcorr(0, ix, iy) = fwk_y_view(1, iy); });
        });

    // zbcmin=0;
    // zbcmax=0;
    // for (int iy = 0; iy < ny; ++iy) {
    //     for (int ix = 0; ix < nx; ++ix)
    //         fwk_x(0, ix) = fcorr[0][ix][iy];
    //     CubicSplineInterpolator<T, MemorySpace>::mkspline(x, nx, fwk_x,
    //     fwk4_x, ibcxmin, 0.0, ibcxmax, 0.0, wk_x); if (ier != 0) return; for
    //     (int ix = 0; ix < nx; ++ix)
    //         fcorr[1][ix][iy] = fwk_x(1, ix);
    // }
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<execution_space>(ny, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type &team) {
          const int iy = team.league_rank();
          auto fwk_x_view = Rank2View<T, MemorySpace>(
              fspl_l_x.data_handle() + 2 * nx * iy, 2, nx);
          auto wk_x_view =
              Rank1View<T, MemorySpace>(wk_l.data_handle() + nx * iy, nx);
          auto fwk4_x_view = Rank2View<T, MemorySpace>(
              fwk4_l_x.data_handle() + 4 * nx * iy, 4, nx);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, nx),
              [=](int ix) { fwk_x_view(0, ix) = fcorr(0, ix, iy); });

          if (team.team_rank() == 0) {
            CubicSplineInterpolator<T, MemorySpace>::mkspline(
                x, nx, fwk_x_view, fwk4_x_view, ibcxmin, zbcmin, ibcxmax,
                zbcmax, wk_x_view);
          }

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, nx),
              [=](int ix) { fcorr(1, ix, iy) = fwk_x_view(1, ix); });
        });

    // for (int c = 0; c < 2; ++c)
    //     for (int ix = 0; ix < nx; ++ix)
    //         for (int iy = 0; iy < ny; ++iy)
    //             f(c + 2, ix, iy) += fcorr[c][ix][iy];
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(const int ix, const int iy) {
          f(2, ix, iy) += fcorr(0, ix, iy);
          f(3, ix, iy) += fcorr(1, ix, iy);
        });
  }
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void BiCubicSplineInterpolator<T, MemorySpace>::herm2xy(
    double xget, double yget, Rank1View<T, MemorySpace> x, int &nx,
    Rank1View<T, MemorySpace> y, int &ny, int &i, int &j, double &xparam,
    double &yparam, double &hx, double &hxi, double &hy, double &hyi,
    int &ier) {
  ier = 0;

  double zxget = xget;
  double zyget = yget;

  const double zxtol = 4.0e-7 * std::max(std::fabs(x[0]), std::fabs(x[nx - 1]));
  const double zytol = 4.0e-7 * std::max(std::fabs(y[0]), std::fabs(y[ny - 1]));

  // X range check / fixup
  if (xget < x[0] - zxtol || xget > x[nx - 1] + zxtol) {
    ier = 1;
    printf(
        "?herm2xy:  xget = %g out of range %g to %g\n", xget, x[0], x[nx - 1]);
    return;
  } else if (xget < x[0]) {
    if (xget < x[0] - 0.5 * zxtol || xget > x[nx - 1] + 0.5 * zxtol)
      printf("herm2xy:  xget = %g beyond range %g to %g (fixup applied)\n",
             xget, x[0], x[nx - 1]);
    zxget = x[0];
  } else if (xget > x[nx - 1]) {
    if (xget < x[0] - 0.5 * zxtol || xget > x[nx - 1] + 0.5 * zxtol)
      printf("herm2xy:  xget = %g beyond range %g to %g (fixup applied)\n",
             xget, x[0], x[nx - 1]);
    zxget = x[nx - 1];
  }

  // Y range check / fixup
  if (yget < y[0] - zytol || yget > y[ny - 1] + zytol) {
    ier = 1;
    printf("?herm2xy:  yget = %g out of range %g to %g\n", yget, y[0],
           y[ny - 1]);
    return;
  } else if (yget < y[0]) {
    if (yget < y[0] - 0.5 * zytol || yget > y[ny - 1] + 0.5 * zytol)
    printf("herm2xy:  yget = %g beyond range %g to %g (fixup applied)\n", yget,
           y[0], y[ny - 1]);
    zyget = y[0];
  } else if (yget > y[ny - 1]) {
    if (yget < y[0] - 0.5 * zytol || yget > y[ny - 1] + 0.5 * zytol)
    printf("herm2xy:  yget = %g beyond range %g to %g (fixup applied)\n", yget,
           y[0], y[ny - 1]);
    zyget = y[ny - 1];
  }

  int nxm = nx - 1; // Number of intervals in x
  int nym = ny - 1; // Number of intervals in y
  // Determine zone index i
  int ii = static_cast<int>(nxm * (zxget - x[0]) / (x[nx - 1] - x[0]));
  i = std::min(nxm - 1, ii);
  if (zxget < x[i])
    --i;
  else if (zxget > x[i + 1])
    ++i;

  // Determine zone index j
  ii = static_cast<int>(nym * (zyget - y[0]) / (y[ny - 1] - y[0]));
  j = std::min(nym - 1, ii);
  if (zyget < y[j])
    --j;
  else if (zyget > y[j + 1])
    ++j;

  hx = x[i + 1] - x[i];
  hy = y[j + 1] - y[j];

  hxi = 1.0 / hx;
  hyi = 1.0 / hy;

  xparam = (zxget - x[i]) * hxi;
  yparam = (zyget - y[j]) * hyi;
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void BiCubicSplineInterpolator<T, MemorySpace>::fvbicub(
    Kokkos::View<int*, MemorySpace> ict, int ivec, int ivecd,
    Rank1View<T, MemorySpace> fval, const int &i, const int &j,
    const double &xparam, const double &yparam, const double &hx,
    const double &hxi, const double &hy, const double &hyi) {
  constexpr double sixth = 1.0 / 6.0;
  const double z36th = sixth * sixth;
  int iadr = 0;

  /* ------------------------------------------------------------------
   * ict[0] = 1 or 2  →   f, df/dx, df/dy, d²f/dx², d²f/dy², d²f/dxdy
   * ----------------------------------------------------------------*/
  if (ict[0] <= 2) {
    /********** f (function value) **********/
    if (ict[0] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double xp2 = xp * xp, xpi2 = xpi * xpi;
      double cx = xp * (xp2 - 1.0);
      double cxi = xpi * (xpi2 - 1.0);
      double hx2 = hx * hx;

      double yp = yparam, ypi = 1.0 - yp;
      double yp2 = yp * yp, ypi2 = ypi * ypi;
      double cy = yp * (yp2 - 1.0);
      double cyi = ypi * (ypi2 - 1.0);
      double hy2 = hy * hy;

      double sum = xpi * (ypi * f_(0, i, j) + yp * f_(0, i, j + 1)) +
                   xp * (ypi * f_(0, i + 1, j) + yp * f_(0, i + 1, j + 1));
      sum += sixth * hx2 *
             (cxi * (ypi * f_(1, i, j) + yp * f_(1, i, j + 1)) +
              cx * (ypi * f_(1, i + 1, j) + yp * f_(1, i + 1, j + 1)));
      sum += sixth * hy2 *
             (xpi * (cyi * f_(2, i, j) + cy * f_(2, i, j + 1)) +
              xp * (cyi * f_(2, i + 1, j) + cy * f_(2, i + 1, j + 1)));
      sum += z36th * hx2 * hy2 *
             (cxi * (cyi * f_(3, i, j) + cy * f_(3, i, j + 1)) +
              cx * (cyi * f_(3, i + 1, j) + cy * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** df/dx **********/
    if (ict[1] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double xp2 = xp * xp, xpi2 = xpi * xpi;
      double cxd = 3.0 * xp2 - 1.0;
      double cxdi = -3.0 * xpi2 + 1.0;
      double yp = yparam, ypi = 1.0 - yp;
      double yp2 = yp * yp, ypi2 = ypi * ypi;
      double cy = yp * (yp2 - 1.0);
      double cyi = ypi * (ypi2 - 1.0);
      double hy2 = hy * hy;

      double sum = hxi * (-(ypi * f_(0, i, j) + yp * f_(0, i, j + 1)) +
                          (ypi * f_(0, i + 1, j) + yp * f_(0, i + 1, j + 1)));
      sum += sixth * hx *
             (cxdi * (ypi * f_(1, i, j) + yp * f_(1, i, j + 1)) +
              cxd * (ypi * f_(1, i + 1, j) + yp * f_(1, i + 1, j + 1)));
      sum += sixth * hxi * hy2 *
             (-(cyi * f_(2, i, j) + cy * f_(2, i, j + 1)) +
              (cyi * f_(2, i + 1, j) + cy * f_(2, i + 1, j + 1)));
      sum += z36th * hx * hy2 *
             (cxdi * (cyi * f_(3, i, j) + cy * f_(3, i, j + 1)) +
              cxd * (cyi * f_(3, i + 1, j) + cy * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** df/dy **********/
    if (ict[2] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double xp2 = xp * xp, xpi2 = xpi * xpi;
      double cx = xp * (xp2 - 1.0);
      double cxi = xpi * (xpi2 - 1.0);
      double hx2 = hx * hx;
      double yp = yparam, ypi = 1.0 - yp;
      double yp2 = yp * yp, ypi2 = ypi * ypi;
      double cyd = 3.0 * yp2 - 1.0;
      double cydi = -3.0 * ypi2 + 1.0;

      double sum = hyi * (xpi * (-f_(0, i, j) + f_(0, i, j + 1)) +
                          xp * (-f_(0, i + 1, j) + f_(0, i + 1, j + 1)));
      sum += sixth * hx2 * hyi *
             (cxi * (-f_(1, i, j) + f_(1, i, j + 1)) +
              cx * (-f_(1, i + 1, j) + f_(1, i + 1, j + 1)));
      sum += sixth * hy *
             (xpi * (cydi * f_(2, i, j) + cyd * f_(2, i, j + 1)) +
              xp * (cydi * f_(2, i + 1, j) + cyd * f_(2, i + 1, j + 1)));
      sum += z36th * hx2 * hy *
             (cxi * (cydi * f_(3, i, j) + cyd * f_(3, i, j + 1)) +
              cx * (cydi * f_(3, i + 1, j) + cyd * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d²f/dx² **********/
    if (ict[3] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double yp = yparam, ypi = 1.0 - yp;
      double yp2 = yp * yp, ypi2 = ypi * ypi;
      double cy = yp * (yp2 - 1.0);
      double cyi = ypi * (ypi2 - 1.0);
      double hy2 = hy * hy;

      double sum = xpi * (ypi * f_(1, i, j) + yp * f_(1, i, j + 1)) +
                   xp * (ypi * f_(1, i + 1, j) + yp * f_(1, i + 1, j + 1));
      sum += sixth * hy2 *
             (xpi * (cyi * f_(3, i, j) + cy * f_(3, i, j + 1)) +
              xp * (cyi * f_(3, i + 1, j) + cy * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d²f/dy² **********/
    if (ict[4] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double xp2 = xp * xp, xpi2 = xpi * xpi;
      double cx = xp * (xp2 - 1.0);
      double cxi = xpi * (xpi2 - 1.0);
      double hx2 = hx * hx;
      double yp = yparam, ypi = 1.0 - yp;

      double sum = xpi * (ypi * f_(2, i, j) + yp * f_(2, i, j + 1)) +
                   xp * (ypi * f_(2, i + 1, j) + yp * f_(2, i + 1, j + 1));
      sum += sixth * hx2 *
             (cxi * (ypi * f_(3, i, j) + yp * f_(3, i, j + 1)) +
              cx * (ypi * f_(3, i + 1, j) + yp * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d²f/dxdy **********/
    if (ict[5] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double xp2 = xp * xp, xpi2 = xpi * xpi;
      double cxd = 3.0 * xp2 - 1.0;
      double cxdi = -3.0 * xpi2 + 1.0;
      double yp = yparam, ypi = 1.0 - yp;
      double yp2 = yp * yp, ypi2 = ypi * ypi;
      double cyd = 3.0 * yp2 - 1.0;
      double cydi = -3.0 * ypi2 + 1.0;

      double sum = hxi * hyi *
                   (f_(0, i, j) - f_(0, i, j + 1) - f_(0, i + 1, j) +
                    f_(0, i + 1, j + 1));
      sum += sixth * hx * hyi *
             (cxdi * (-f_(1, i, j) + f_(1, i, j + 1)) +
              cxd * (-f_(1, i + 1, j) + f_(1, i + 1, j + 1)));
      sum += sixth * hxi * hy *
             (-(cydi * f_(2, i, j) + cyd * f_(2, i, j + 1)) +
              (cydi * f_(2, i + 1, j) + cyd * f_(2, i + 1, j + 1)));
      sum += z36th * hx * hy *
             (cxdi * (cydi * f_(3, i, j) + cyd * f_(3, i, j + 1)) +
              cxd * (cydi * f_(3, i + 1, j) + cyd * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }
  }

  /* ------------------------------------------------------------------
   * ict[0] = 3  → 3rd‑order derivative combinations
   * ----------------------------------------------------------------*/
  else if (ict[0] == 3) {
    /********** d³f/dx³ **********/
    if (ict[1] == 1) {
      iadr++;
      double yp = yparam, ypi = 1.0 - yp;
      double yp2 = yp * yp, ypi2 = ypi * ypi;
      double cy = yp * (yp2 - 1.0);
      double cyi = ypi * (ypi2 - 1.0);
      double hy2 = hy * hy;
      double sum = hxi * (-(ypi * f_(1, i, j) + yp * f_(1, i, j + 1)) +
                          (ypi * f_(1, i + 1, j) + yp * f_(1, i + 1, j + 1)));
      sum += sixth * hy2 * hxi *
             (-(cyi * f_(3, i, j) + cy * f_(3, i, j + 1)) +
              (cyi * f_(3, i + 1, j) + cy * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d³f/dx²dy **********/
    if (ict[2] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double yp = yparam, ypi = 1.0 - yp;
      double yp2 = yp * yp, ypi2 = ypi * ypi;
      double cyd = 3.0 * yp2 - 1.0;
      double cydi = -3.0 * ypi2 + 1.0;
      double sum = hyi * (xpi * (-f_(1, i, j) + f_(1, i, j + 1)) +
                          xp * (-f_(1, i + 1, j) + f_(1, i + 1, j + 1)));
      sum += sixth * hy *
             (xpi * (cydi * f_(3, i, j) + cyd * f_(3, i, j + 1)) +
              xp * (cydi * f_(3, i + 1, j) + cyd * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d³f/dxdy² **********/
    if (ict[3] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double xp2 = xp * xp, xpi2 = xpi * xpi;
      double cxd = 3.0 * xp2 - 1.0;
      double cxdi = -3.0 * xpi2 + 1.0;
      double yp = yparam, ypi = 1.0 - yp;
      double sum = hxi * (-(ypi * f_(2, i, j) + yp * f_(2, i, j + 1)) +
                          (ypi * f_(2, i + 1, j) + yp * f_(2, i + 1, j + 1)));
      sum += sixth * hx *
             (cxdi * (ypi * f_(3, i, j) + yp * f_(3, i, j + 1)) +
              cxd * (ypi * f_(3, i + 1, j) + yp * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d³f/dy³ **********/
    if (ict[4] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double xp2 = xp * xp, xpi2 = xpi * xpi;
      double cx = xp * (xp2 - 1.0);
      double cxi = xpi * (xpi2 - 1.0);
      double hx2 = hx * hx;
      double sum = hyi * (xpi * (-f_(2, i, j) + f_(2, i, j + 1)) +
                          xp * (-f_(2, i + 1, j) + f_(2, i + 1, j + 1)));
      sum += sixth * hx2 * hyi *
             (cxi * (-f_(3, i, j) + f_(3, i, j + 1)) +
              cx * (-f_(3, i + 1, j) + f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }
  }

  /* ------------------------------------------------------------------
   * ict[0] = 4  → 4th‑order derivative combinations
   * ----------------------------------------------------------------*/
  else if (ict[0] == 4) {
    /********** d⁴f/dx³dy **********/
    if (ict[1] == 1) {
      iadr++;
      double yp = yparam, ypi = 1.0 - yp;
      double yp2 = yp * yp, ypi2 = ypi * ypi;
      double cyd = 3.0 * yp2 - 1.0;
      double cydi = -3.0 * ypi2 + 1.0;
      double sum = hxi * hyi *
                   (f_(1, i, j) - f_(1, i, j + 1) - f_(1, i + 1, j) +
                    f_(1, i + 1, j + 1));
      sum += sixth * hy * hxi *
             (-(cydi * f_(3, i, j) + cyd * f_(3, i, j + 1)) +
              (cydi * f_(3, i + 1, j) + cyd * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d⁴f/dx²dy² **********/
    if (ict[2] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double yp = yparam, ypi = 1.0 - yp;
      double sum = xpi * (ypi * f_(3, i, j) + yp * f_(3, i, j + 1)) +
                   xp * (ypi * f_(3, i + 1, j) + yp * f_(3, i + 1, j + 1));
      fval(iadr - 1) = sum;
    }

    /********** d⁴f/dxdy³ **********/
    if (ict[3] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double xp2 = xp * xp, xpi2 = xpi * xpi;
      double cxd = 3.0 * xp2 - 1.0;
      double cxdi = -3.0 * xpi2 + 1.0;
      double sum = hxi * hyi *
                   (f_(2, i, j) - f_(2, i, j + 1) - f_(2, i + 1, j) +
                    f_(2, i + 1, j + 1));
      sum += sixth * hx * hyi *
             (cxdi * (-f_(3, i, j) + f_(3, i, j + 1)) +
              cxd * (-f_(3, i + 1, j) + f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }
  }

  /* ------------------------------------------------------------------
   * ict[0] = 5  → 5th‑order derivative combinations
   * ----------------------------------------------------------------*/
  else if (ict[0] == 5) {
    /********** d⁵f/dx³dy² **********/
    if (ict[1] == 1) {
      iadr++;
      double yp = yparam, ypi = 1.0 - yp;
      double sum = hxi * (-(ypi * f_(3, i, j) + yp * f_(3, i, j + 1)) +
                          (ypi * f_(3, i + 1, j) + yp * f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d⁵f/dx²dy³ **********/
    if (ict[2] == 1) {
      iadr++;
      double xp = xparam, xpi = 1.0 - xp;
      double sum = hyi * (xpi * (-f_(3, i, j) + f_(3, i, j + 1)) +
                          xp * (-f_(3, i + 1, j) + f_(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }
  }

  /* ------------------------------------------------------------------
   * ict[0] = 6  → 6th‑order derivative (dx³dy³)
   * ----------------------------------------------------------------*/
  else if (ict[0] == 6) {
    iadr++;
    double sum =
        hxi * hyi *
        (f_(3, i, j) - f_(3, i, j + 1) - f_(3, i + 1, j) + f_(3, i + 1, j + 1));
    fval(iadr - 1) = sum;
  }
}

template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::evbicub(
    double xget, double yget,
    Kokkos::View<int*, MemorySpace> ict,
    Rank1View<T, MemorySpace> fval, // output (size depends on ict)
    int &ier) {
  // Local variables
  int i = 0, j = 0;
  T xparam = 0.0, yparam = 0.0;
  T hx = 0.0, hy = 0.0;
  T hxi = 0.0, hyi = 0.0;

  // Call herm2xy to locate cell and compute params
  herm2xy(xget, yget, x_, nx_, y_, ny_, i, j, xparam, yparam, hx, hxi, hy, hyi,
          ier);

  if (ier != 0)
    return;

  // Call fvbicub with scalar-vector interface
  fvbicub(ict, 1, 1, fval, i, j, xparam, yparam, hx, hxi, hy, hyi);
}

} // namespace pcms
#endif