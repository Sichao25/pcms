
#ifndef MLS_RBF_OPTIONS_HPP
#define MLS_RBF_OPTIONS_HPP

#include "mdspan/mdspan.hpp"
#include "pcms/arrays.h"
#include "pcms/assert.h"
#include "pcms/memory_spaces.h"
#include "pcms/types.h"
#include <Kokkos_Core.hpp>
#include <cmath>

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

void ibc_ck(int ibc, const std::string &slbl, const std::string &xlbl, int imin,
            int imax) {

  PCMS_ALWAYS_ASSERT(ibc >= imin && ibc <= imax);
}

// TODO: better error handling, use pcms existing error handling
// TODO: better documentation
// TODO: add more checks/assetions for input parameters
template <typename T, typename MemorySpace> class CubicSplineInterpolator {
public:
  CubicSplineInterpolator() = default;
  using execution_space = typename MemorySpace::execution_space;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;
  Rank1View<T, MemorySpace> x_;
  int nx_;
  static void splinck(Rank1View<T, MemorySpace> x, const T &ztol);

  static KOKKOS_INLINE_FUNCTION void range_check(T &xget, T &zxget, Rank1View<T, MemorySpace> x, const int &nx, int &ier);

  static KOKKOS_FUNCTION void v_spline(const int &k_bc1, const int &k_bcn,
                                       const int &n,
                                       Rank1View<T, MemorySpace> x,
                                       Rank2View<T, MemorySpace> f,
                                       Rank1View<T, MemorySpace> wk);

}; // end of cubic spline class

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::splinck(
    Rank1View<T, MemorySpace> x, const T &ztol) {
  int inx = static_cast<int>(x.extent(0));

  if (inx <= 1)
    return;

  // T dxavg = (x[inx - 1] - x[0]) / (inx - 1);
  // T zeps = std::abs(ztol * dxavg);

  // for (int ix = 1; ix < inx; ++ix) {
  //   T zdiffx = x[ix] - x[ix - 1];
  //   PCMS_ALWAYS_ASSERT(zdiffx > 0.0);
  //   T zdiff = zdiffx - dxavg;
  //   PCMS_ALWAYS_ASSERT(std::abs(zdiff) <= zeps);
  // }
  T dxavg = 0.0;
  T zeps = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<execution_space>(0, 1),
      KOKKOS_LAMBDA(const int, T &avg, T &eps) {
        avg = (x[inx - 1] - x[0]) / (inx - 1);
        eps = std::abs(ztol * avg);
      },
      dxavg, zeps);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(1, inx),
      KOKKOS_LAMBDA(const int ix) {
        T zdiffx = x(ix) - x(ix - 1);
        assert(zdiffx > 0.0);
        T zdiff = zdiffx - dxavg;
        assert(std::abs(zdiff) <= zeps);
      });
}

template <typename T, typename MemorySpace>
class ExplicitCubicSplineInterpolator
    : public CubicSplineInterpolator<T, MemorySpace> {
public:
  using typename CubicSplineInterpolator<T, MemorySpace>::execution_space;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;
  ExplicitCubicSplineInterpolator() = default;
  ExplicitCubicSplineInterpolator(Rank1View<T, MemorySpace> x, const int &nx,
                                  Rank2View<T, MemorySpace> fspl,
                                  const int &ibcxmin, const T &bcxmin,
                                  const int &ibcxmax, const T &bcxmax,
                                  Rank1View<T, MemorySpace> wk);
  Rank2View<T, MemorySpace> fspl_;
  void setup(Rank1View<T, MemorySpace> x, const int &nx,
                      Rank2View<T, MemorySpace> fspl, const int &ibcxmin,
                      const T &bcxmin, const int &ibcxmax,
                      const T &bcxmax, Rank1View<T, MemorySpace> wk);
  static KOKKOS_FUNCTION void cspline(Rank1View<T, MemorySpace> x,
                                      const int &nx,
                                      Rank2View<T, MemorySpace> fspl,
                                      const int &ibcxmin, const T &bcxmin,
                                      const int &ibcxmax, const T &bcxmax,
                                      Rank1View<T, MemorySpace> wk);
  static KOKKOS_INLINE_FUNCTION void cspevfn(Kokkos::View<int *, MemorySpace> selector,
                                      Rank1View<T, MemorySpace> fval,
                                      const int &i, const T &dx,
                                      Rank2View<T, MemorySpace> fspl);

  static KOKKOS_INLINE_FUNCTION void
  cspeval(T xget, Kokkos::View<int *, MemorySpace> iselect,
          Rank1View<T, MemorySpace> fval, Rank1View<T, MemorySpace> x,
          const int &nx, Rank2View<T, MemorySpace> fspl, int &ier);

  static KOKKOS_INLINE_FUNCTION void cspevx(T xget, Rank1View<T, MemorySpace> x,
                                     const int &nx, int &i, T &dx,
                                     int &ier);
  void evaluate(Kokkos::View<int *, MemorySpace> selector,
                         Rank1View<T, MemorySpace> xvec,
                         Rank2View<T, MemorySpace> fval);
};

template <typename T, typename MemorySpace>
class CompactCubicSplineInterpolator
    : public ExplicitCubicSplineInterpolator<T, MemorySpace> {
public:
  using typename CubicSplineInterpolator<T, MemorySpace>::execution_space;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;
  Rank2View<T, MemorySpace> fs2_;
  CompactCubicSplineInterpolator() = default;
  CompactCubicSplineInterpolator(Rank1View<T, MemorySpace> x, const int &nx,
                                 Rank2View<T, MemorySpace> fspl,
                                 Rank2View<T, MemorySpace> fspl4,
                                 const int &ibcxmin, const T &bcxmin,
                                 const int &ibcxmax, const T &bcxmax,
                                 Rank1View<T, MemorySpace> wk);
  void setup(Rank1View<T, MemorySpace> x, const int &nx,
                     Rank2View<T, MemorySpace> fspl,
                     Rank2View<T, MemorySpace> fspl4, const int &ibcxmin,
                     const T &bcxmin, const int &ibcxmax,
                     const T &bcxmax, Rank1View<T, MemorySpace> wk);
  static KOKKOS_FUNCTION void
  mkspline(Rank1View<T, MemorySpace> x, const int &nx,
           Rank2View<T, MemorySpace> fspl, Rank2View<T, MemorySpace> fspl4,
           const int &ibcxmin, const T &bcxmin, const int &ibcxmax,
           const T &bcxmax, Rank1View<T, MemorySpace> wk);

  static KOKKOS_INLINE_FUNCTION void
  fvspline(Kokkos::View<int *, MemorySpace> selector,
           Rank1View<T, MemorySpace> fval, const int &i, const T &xparam,
           const T &hx, const T &hxi, Rank2View<T, MemorySpace> fs2);

  static KOKKOS_INLINE_FUNCTION void
  evspline(T xget, Kokkos::View<int *, MemorySpace> ict,
           Rank1View<T, MemorySpace> fval, Rank1View<T, MemorySpace> x,
           const int &nx, Rank2View<T, MemorySpace> fs2, int &ier);

  static KOKKOS_INLINE_FUNCTION void herm1x(T xget, Rank1View<T, MemorySpace> x,
                                     const int &nx, int &i, T &xparam,
                                     T &hx, T &hxi, int &ier);

  void evaluate(Kokkos::View<int *, MemorySpace> selector,
                        Rank1View<T, MemorySpace> xvec,
                        Rank2View<T, MemorySpace> fval);
};

template <typename T, typename MemorySpace>
ExplicitCubicSplineInterpolator<T, MemorySpace>::
    ExplicitCubicSplineInterpolator(Rank1View<T, MemorySpace> x, const int &nx,
                                    Rank2View<T, MemorySpace> fspl,
                                    const int &ibcxmin, const T &bcxmin,
                                    const int &ibcxmax, const T &bcxmax,
                                    Rank1View<T, MemorySpace> wk) {
  PCMS_ALWAYS_ASSERT(x.extent(0) >= 2);
  this->nx_ = x.extent(0);
  this->x_ = x;
  setup(x, nx, fspl, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);
  this->fspl_ = fspl;
}

template <typename T, typename MemorySpace>
void ExplicitCubicSplineInterpolator<T, MemorySpace>::setup(
    Rank1View<T, MemorySpace> x, const int &nx, Rank2View<T, MemorySpace> fspl,
    const int &ibcxmin, const T &bcxmin, const int &ibcxmax,
    const T &bcxmax, Rank1View<T, MemorySpace> wk) {
  // Kokkos::parallel_for("explicit cubic",
  // Kokkos::TeamPolicy<execution_space>(1, Kokkos::AUTO), KOKKOS_LAMBDA(const
  // member_type& team_member) {
  //     Kokkos::single(Kokkos::PerTeam(team_member), [=]() {
  //         cspline(x, nx, fspl, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);
  //     });
  // });
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(0, 1), KOKKOS_LAMBDA(const int i) {
        cspline(x, nx, fspl, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);
      });
}

template <typename T, typename MemorySpace>
CompactCubicSplineInterpolator<T, MemorySpace>::CompactCubicSplineInterpolator(
    Rank1View<T, MemorySpace> x, const int &nx, Rank2View<T, MemorySpace> fspl,
    Rank2View<T, MemorySpace> fspl4, const int &ibcxmin, const T &bcxmin,
    const int &ibcxmax, const T &bcxmax, Rank1View<T, MemorySpace> wk) {
  this->nx_ = x.extent(0);
  this->x_ = x;
  setup(x, nx, fspl, fspl4, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);
  this->fs2_ = fspl;
}

template <typename T, typename MemorySpace>
void CompactCubicSplineInterpolator<T, MemorySpace>::setup(
    Rank1View<T, MemorySpace> x, const int &nx, Rank2View<T, MemorySpace> fspl,
    Rank2View<T, MemorySpace> fspl4, const int &ibcxmin, const T &bcxmin,
    const int &ibcxmax, const T &bcxmax, Rank1View<T, MemorySpace> wk) {
  Kokkos::parallel_for(
      "compact cubic", Kokkos::TeamPolicy<execution_space>(1, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &team_member) {
        Kokkos::single(Kokkos::PerTeam(team_member), [=]() {
          mkspline(x, nx, fspl, fspl4, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);
        });
      });
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void ExplicitCubicSplineInterpolator<T, MemorySpace>::cspline(
    Rank1View<T, MemorySpace> x, const int &nx, Rank2View<T, MemorySpace> fspl,
    const int &ibcxmin, const T &bcxmin, const int &ibcxmax,
    const T &bcxmax, Rank1View<T, MemorySpace> wk) {

  T half = 0.5;
  T sixth = 0.166666666666666667;

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
  CubicSplineInterpolator<T, MemorySpace>::v_spline(ibcxmin, ibcxmax, nx, x,
                                                    fspl, wk);
  for (int i = 0; i < nx; ++i) {
    fspl(2, i) = half * fspl(2, i);
    fspl(3, i) = sixth * fspl(3, i);
  }
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void CubicSplineInterpolator<T, MemorySpace>::v_spline(
    const int &k_bc1, const int &k_bcn, const int &n,
    Rank1View<T, MemorySpace> x, Rank2View<T, MemorySpace> f,
    Rank1View<T, MemorySpace> wk) {
  int i_bc1 = k_bc1;
  int i_bcn = k_bcn;
  int iord1, iord2, imin, imax;
  T a1, b1, an, bn, f0, fh, h;

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
      T a1 = f(1, 0), an = f(1, 1);
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
      T a1 = f(1, 0), bn = f(2, 1);
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
      T b1 = f(2, 0), an = f(1, 1);
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
      T b1 = f(2, 0), bn = f(2, 1);
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
    T h1 = x[1] - x[0];
    T h2 = x[2] - x[1];
    T h = h1 + h2;

    T dels = (f(0, 2) - f(0, 1)) / h2 - (f(0, 1) - f(0, 0)) / h1;

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
    T h1 = x[1] - x[0];
    T h2 = x[2] - x[1];
    T h = h1 + h2;

    T f1 = f(0, 0) - f(0, 1);
    T f2 = f(0, 2) - f(0, 1);

    // Solve quadratic through 3 points centered at x[1]
    T aa = (f2 * h1 + f1 * h2) / (h1 * h2 * h);
    T bb = (f2 * h1 * h1 - f1 * h2 * h2) / (h1 * h2 * h);

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
      T h2 = x[1] - x[0];
      T h3 = x[2] - x[0];

      T f2 = f(0, 1) - f(0, 0);
      T f3 = f(0, 2) - f(0, 0);

      T aa = a1 / (h2 * h3) + f3 / (h3 * h3 * (h3 - h2)) -
                  f2 / (h2 * h2 * (h3 - h2));
      T bb = -a1 * (h3 * h3 - h2 * h2) / (h2 * h3 * (h3 - h2)) +
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
      T h2 = x[1] - x[0];
      T h3 = x[2] - x[0];

      T f2 = f(0, 1) - f(0, 0);
      T f3 = f(0, 2) - f(0, 0);

      T aa = -(b1 / 2.0) * (h3 - h2) / (h3 * h3 - h2 * h2) -
                  f2 / (h2 * (h3 * h3 - h2 * h2)) +
                  f3 / (h3 * (h3 * h3 - h2 * h2));
      T bb = -(b1 / 2.0) * h2 * h3 * (h3 - h2) / (h3 * h3 - h2 * h2) +
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
      T h2 = x[1] - x[2];
      T h3 = x[0] - x[2];

      T f2 = f(0, 1) - f(0, 2);
      T f3 = f(0, 0) - f(0, 2);

      T aa = an / (h2 * h3) + f3 / (h3 * h3 * (h3 - h2)) -
                  f2 / (h2 * h2 * (h3 - h2));
      T bb = -an * (h3 * h3 - h2 * h2) / (h2 * h3 * (h3 - h2)) +
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
      T h2 = x[1] - x[2];
      T h3 = x[0] - x[2];

      T f2 = f(0, 1) - f(0, 2);
      T f3 = f(0, 0) - f(0, 2);

      T aa = -(bn / 2.0) * (h3 - h2) / (h3 * h3 - h2 * h2) -
                  f2 / (h2 * (h3 * h3 - h2 * h2)) +
                  f3 / (h3 * (h3 * h3 - h2 * h2));
      T bb = -(bn / 2.0) * h2 * h3 * (h3 - h2) / (h3 * h3 - h2 * h2) +
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

    T elem21 = f(3, 0);
    T elemnn1 = f(3, n - 2);

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
        T t = f(3, i - 1) / f(1, i - 1);
        f(1, i) -= t * f(3, i - 1);
        f(2, i) -= t * f(2, i - 1);
        wk[i] -= t * wk[i - 1];

        T q = wk[n - 2] / f(1, i - 1);
        wk[n - 2] = -q * f(3, i - 1);
        f(1, n - 2) -= q * wk[i - 1];
        f(2, n - 2) -= q * f(2, i - 1);
      }

      wk[n - 2] += f(3, n - 3);

      // Complete forward elimination
      T t = wk[n - 2] / f(1, n - 3);
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
        T t;
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

    if (i_bc1 == -1) {
      f(1, n - 1) = f(1, 0);
      f(2, n - 1) = f(2, 0);
      f(3, n - 1) = f(3, 0);
    } else {
      T hn = x[n - 1] - x[n - 2];
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
KOKKOS_INLINE_FUNCTION void CubicSplineInterpolator<T, MemorySpace>::range_check(T &xget, T &zxget, Rank1View<T, MemorySpace> x, const int &nx, int &ier) {
  if (xget < x[0] || xget > x[nx - 1]) {
    T zxtol = 4.0e-7 * std::max(std::abs(x[0]), std::abs(x[nx - 1]));

    if (xget < x[0] - zxtol || xget > x[nx - 1] + zxtol) {
      ier = 1; // Error code for out of range
      printf("cspevx:  xget=%.6f out of range %.6f to %.6f\n", xget, x[0],
             x[nx - 1]);

      return;
    } else {
      printf("cspevx:  xget=%.6f beyond range %.6f to %.6f (fixup applied)\n",
             xget, x[0], x[nx - 1]);
      zxget = (xget < x[0]) ? x[0] : x[nx - 1];
    }
  }
}

template <typename T, typename MemorySpace>
KOKKOS_INLINE_FUNCTION void ExplicitCubicSplineInterpolator<T, MemorySpace>::cspevfn(
    Kokkos::View<int *, MemorySpace> selector, Rank1View<T, MemorySpace> fval,
    const int &i, const T &dx, Rank2View<T, MemorySpace> fspl) {

  int iaval = 0;

  if (selector[0] == 3) {
    // Third derivative only
    iaval++;
    fval(iaval - 1) = 6.0 * fspl(3, i);
  } else {
    if (selector[0] > 0) {
      // Evaluate f
      iaval++;
      fval(iaval - 1) =
          fspl(0, i) + dx * (fspl(1, i) + dx * (fspl(2, i) + dx * fspl(3, i)));
    }

    if (selector[1] > 0) {
      // Evaluate df/dx
      iaval++;
      fval(iaval - 1) =
          fspl(1, i) + dx * (2.0 * fspl(2, i) + dx * 3.0 * fspl(3, i));
    }

    if (selector[2] > 0) {
      // Evaluate d2f/dx2
      iaval++;
      fval(iaval - 1) = 2.0 * fspl(2, i) + dx * 6.0 * fspl(3, i);
    }
  }
} // end cspevfn

template <typename T, typename MemorySpace>
KOKKOS_INLINE_FUNCTION void ExplicitCubicSplineInterpolator<T, MemorySpace>::cspeval(
    T xget, Kokkos::View<int *, MemorySpace> iselect,
    Rank1View<T, MemorySpace> fval, Rank1View<T, MemorySpace> x, const int &nx,
    Rank2View<T, MemorySpace> fspl, int &ier) {
  int ia = 0;
  T dxa = 0.0;
  cspevx(xget, x, nx, ia, dxa, ier);
  if (ier != 0) {
    printf("cspeval: error in cspevx, ier = %d\n", ier);
    return;
  }

  cspevfn(iselect, fval, ia, dxa, fspl);
}

template <typename T, typename MemorySpace>
KOKKOS_INLINE_FUNCTION void ExplicitCubicSplineInterpolator<T, MemorySpace>::cspevx(
    T xget, Rank1View<T, MemorySpace> x, const int &nx, int &i, T &dx,
    int &ier) {
  int nxm = nx - 1;
  T zxget = xget;

  // Range check
  CubicSplineInterpolator<T, MemorySpace>::range_check(xget, zxget, x, nx, ier);

  int ii = static_cast<int>(nxm * (zxget - x[0]) / (x[nx - 1] - x[0]));
  i = std::min(nxm - 1, ii); // Fortran is 1-based, C++ is 0-based

  if (zxget < x[i]) {
    i = std::max(0, i - 1);
  } else if (zxget > x[i + 1]) {
    i = std::min(nxm - 1, i + 1);
  }

  dx = zxget - x[i];
}

template <typename T, typename MemorySpace>
KOKKOS_FUNCTION void CompactCubicSplineInterpolator<T, MemorySpace>::mkspline(
    Rank1View<T, MemorySpace> x, const int &nx, Rank2View<T, MemorySpace> fspl,
    Rank2View<T, MemorySpace> fspl4, const int &ibcxmin, const T &bcxmin,
    const int &ibcxmax, const T &bcxmax, Rank1View<T, MemorySpace> wk) {
  // TODO: input check, size check and ascending

  // Copy f data to fspl4 and zero out second derivative output
  for (int i = 0; i < nx; ++i) {
    fspl4(0, i) = fspl(0, i);
    fspl(1, i) = 0.0;
  }

  // Call traditional spline generator
  ExplicitCubicSplineInterpolator<T, MemorySpace>::cspline(
      x, nx, fspl4, ibcxmin, bcxmin, ibcxmax, bcxmax, wk);

  for (int i = 0; i < nx - 1; ++i) {
    fspl(1, i) = 2.0 * fspl4(2, i);
  }

  fspl(1, nx - 1) =
      2.0 * fspl4(2, nx - 2) + (x[nx - 1] - x[nx - 2]) * 6.0 * fspl4(3, nx - 2);
}

template <typename T, typename MemorySpace>
void ExplicitCubicSplineInterpolator<T, MemorySpace>::evaluate(
    const Kokkos::View<int *, MemorySpace> selector,
    Rank1View<T, MemorySpace> xvec, Rank2View<T, MemorySpace> fval) {
  int ivd = fval.extent(1);
  // for (int i = 0; i < xvec.extent(0); ++i){
  //   int ier = 0;
  //   auto fval_view = Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd,
  //   ivd); cspeval(xvec(i), selector, fval_view, x_, nx_, fspl_, ier);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(0, xvec.extent(0)),
      KOKKOS_CLASS_LAMBDA(const int i) {
        int ier = 0;
        auto fval_view =
            Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd, ivd);
        cspeval(xvec(i), selector, fval_view, this->x_, this->nx_, this->fspl_,
                ier);
      });
}

template <typename T, typename MemorySpace>
void CompactCubicSplineInterpolator<T, MemorySpace>::evaluate(
    Kokkos::View<int *, MemorySpace> selector, Rank1View<T, MemorySpace> xvec,
    Rank2View<T, MemorySpace> fval) {
  int ivd = fval.extent(1);
  // for (int i = 0; i < xvec.extent(0); ++i){
  //   int ier = 0;
  //   auto fval_view = Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd,
  //   ivd); evspline(xvec(i), selector, fval_view, x_, nx_, fs2_, ier);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(0, xvec.extent(0)),
      KOKKOS_CLASS_LAMBDA(const int i) {
        int ier = 0;
        auto fval_view =
            Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd, ivd);
        evspline(xvec(i), selector, fval_view, this->x_, this->nx_, this->fs2_,
                 ier);
      });
}

template <typename T, typename MemorySpace>
KOKKOS_INLINE_FUNCTION void CompactCubicSplineInterpolator<T, MemorySpace>::fvspline(
    Kokkos::View<int *, MemorySpace> selector, Rank1View<T, MemorySpace> fval,
    const int &i, const T &xparam, const T &hx, const T &hxi,
    Rank2View<T, MemorySpace> fs2) {
  const T sixth = 1.0 / 6.0;
  int iadr = 0;

  if (selector[0] <= 2) {
    if (selector[0] == 1) {
      // Function value f(x)
      ++iadr;
      // TODO: too much temp varibale created?
      T xp = xparam;
      T xpi = 1.0 - xp;
      T xp2 = xp * xp;
      T xpi2 = xpi * xpi;
      T cx = xp * (xp2 - 1.0);
      T cxi = xpi * (xpi2 - 1.0);
      T hx2 = hx * hx;

      T sum = xpi * fs2(0, i) + xp * fs2(0, i + 1);
      sum += sixth * hx2 * (cxi * fs2(1, i) + cx * fs2(1, i + 1));

      fval(iadr - 1) = sum; // zero-based indexing
    }

    if (selector[1] == 1) {
      // First derivative df/dx
      ++iadr;
      T xp = xparam;
      T xpi = 1.0 - xp;
      T xp2 = xp * xp;
      T xpi2 = xpi * xpi;

      T cxd = 3.0 * xp2 - 1.0;
      T cxdi = -3.0 * xpi2 + 1.0;

      T sum = hxi * (fs2(0, i + 1) - fs2(0, i));
      sum += sixth * hx * (cxdi * fs2(1, i) + cxd * fs2(1, i + 1));

      fval(iadr - 1) = sum;
    }

    if (selector[2] == 1) {
      // Second derivative d2f/dx2
      ++iadr;
      T xp = xparam;
      T xpi = 1.0 - xp;

      T sum = xpi * fs2(1, i) + xp * fs2(1, i + 1);
      fval(iadr - 1) = sum;
    }

  } else {
    // Third derivative d3f/dx3
    iadr = 1;
    fval(iadr - 1) = hxi * (fs2(1, i + 1) - fs2(1, i));
  }
}

template <typename T, typename MemorySpace>
KOKKOS_INLINE_FUNCTION void CompactCubicSplineInterpolator<T, MemorySpace>::herm1x(
    T xget, Rank1View<T, MemorySpace> x, const int &nx, int &i,
    T &xparam, T &hx, T &hxi, int &ier) {
  ier = 0;

  T zxget = xget;

  CubicSplineInterpolator<T, MemorySpace>::range_check(
      xget, zxget, x, nx, ier);

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
KOKKOS_INLINE_FUNCTION void CompactCubicSplineInterpolator<T, MemorySpace>::evspline(
    T xget, Kokkos::View<int *, MemorySpace> ict,
    Rank1View<T, MemorySpace> fval, Rank1View<T, MemorySpace> x,
    const int &nx, // size of x
    Rank2View<T, MemorySpace> fs2, int &ier) {

  // Initialize output zone info
  int i = 0;
  T xparam = 0.0;
  T hx = 0.0;
  T hxi = 0.0;

  // Find the interval containing xget
  herm1x(xget, x, nx, i, xparam, hx, hxi, ier);
  if (ier != 0)
    return;

  // Evaluate spline at the point
  fvspline(ict, fval, i, xparam, hx, hxi, fs2);
}

template <typename T, typename MemorySpace>
class ExplicitBiCubicSplineInterpolator
    : public ExplicitCubicSplineInterpolator<T, MemorySpace> {
public:
  using typename CubicSplineInterpolator<T, MemorySpace>::execution_space;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

  Rank4View<T, MemorySpace> fspl_;
  Rank1View<T, MemorySpace> y_;
  int ny_;

  ExplicitBiCubicSplineInterpolator() = default;

  ExplicitBiCubicSplineInterpolator(
      Rank1View<T, MemorySpace> x,              // size: inx
      int inx, Rank1View<T, MemorySpace> th,    // size: inth
      int inth, Rank4View<T, MemorySpace> fspl, // [4, 4, inx, inth]
      int ibcxmin,
      Rank1View<T, MemorySpace> bcxmin, // size: inth (used if ibcxmin = 1 or 2)
      int ibcxmax,
      Rank1View<T, MemorySpace> bcxmax, // size: inth (used if ibcxmax = 1 or 2)
      int ibcthmin,
      Rank1View<T, MemorySpace>
          bcthmin, // size: inx (used if ibcthmin = 1 or 2)
      int ibcthmax,
      Rank1View<T, MemorySpace>
          bcthmax,                 // size: inx (used if ibcthmax = 1 or 2)
      Rank1View<T, MemorySpace> wk // size: nwk
  );

  static void bcspline(
      Rank1View<T, MemorySpace> x,              // size: inx
      int inx, Rank1View<T, MemorySpace> th,    // size: inth
      int inth, Rank4View<T, MemorySpace> fspl, // [4, 4, inx, inth]
      int ibcxmin,
      Rank1View<T, MemorySpace> bcxmin, // size: inth (used if ibcxmin = 1 or 2)
      int ibcxmax,
      Rank1View<T, MemorySpace> bcxmax, // size: inth (used if ibcxmax = 1 or 2)
      int ibcthmin,
      Rank1View<T, MemorySpace>
          bcthmin, // size: inx (used if ibcthmin = 1 or 2)
      int ibcthmax,
      Rank1View<T, MemorySpace>
          bcthmax,                 // size: inx (used if ibcthmax = 1 or 2)
      Rank1View<T, MemorySpace> wk // size: nwk
  );

  static KOKKOS_INLINE_FUNCTION void
  bcspeval(T xget, T yget, Kokkos::View<int *, MemorySpace> iselect,
           Rank1View<T, MemorySpace> fval, Rank1View<T, MemorySpace> x,
           const int &nx, Rank1View<T, MemorySpace> y, const int &ny,
           Rank4View<T, MemorySpace> fspl, int &ier);

  static KOKKOS_INLINE_FUNCTION void
  bcspevxy(T xget, T yget, Rank1View<T, MemorySpace> x, const int &nx,
           Rank1View<T, MemorySpace> y, const int &ny, int &i, int &j,
           T &dx, T &dy, int &ier);

  static KOKKOS_INLINE_FUNCTION void bcspevfn(
      Kokkos::View<int *, MemorySpace>
          ict, // Selector array for which derivatives to compute
      Rank1View<T, MemorySpace> fval, // Output array: size [ivd, *] (flattened)
      const int &i,                   // Grid cell indices in x direction
      const int &j,                   // Grid cell indices in y direction
      const T &dx,               // x displacements within cells
      const T &dy,               // y displacements within cells
      Rank4View<T, MemorySpace> fspl  // Spline coefficients: [4, 4, nx, ny]
  );

  void evaluate(Kokkos::View<int *, MemorySpace> iselect,
                         Rank1View<T, MemorySpace> xvec,
                         Rank1View<T, MemorySpace> yvec,
                         Rank2View<T, MemorySpace> fval);
};

template <typename T, typename MemorySpace>
class CompactBiCubicSplineInterpolator
    : public CompactCubicSplineInterpolator<T, MemorySpace> {
public:
  using typename CubicSplineInterpolator<T, MemorySpace>::execution_space;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;
  Rank3View<T, MemorySpace> f_;
  Rank1View<T, MemorySpace> y_;
  int ny_;

  CompactBiCubicSplineInterpolator() = default;

  CompactBiCubicSplineInterpolator(
      Rank1View<T, MemorySpace> x, int nx, Rank1View<T, MemorySpace> y, int ny,
      Rank3View<T, MemorySpace> f, int ibcxmin,
      Rank1View<T, MemorySpace> bcxmin, int ibcxmax,
      Rank1View<T, MemorySpace> bcxmax, int ibcymin,
      Rank1View<T, MemorySpace> bcymin, int ibcymax,
      Rank1View<T, MemorySpace> bcymax, Rank1View<T, MemorySpace> wk);

  static void mkbicub(Rank1View<T, MemorySpace> x, int nx,
                      Rank1View<T, MemorySpace> y, int ny,
                      Rank3View<T, MemorySpace> f, int ibcxmin,
                      Rank1View<T, MemorySpace> bcxmin, int ibcxmax,
                      Rank1View<T, MemorySpace> bcxmax, int ibcymin,
                      Rank1View<T, MemorySpace> bcymin, int ibcymax,
                      Rank1View<T, MemorySpace> bcymax,
                      Rank1View<T, MemorySpace> wk);

  static KOKKOS_INLINE_FUNCTION void
  herm2xy(T xget, T yget, Rank1View<T, MemorySpace> x, const int &nx,
          Rank1View<T, MemorySpace> y, const int &ny, int &i, int &j,
          T &xparam, T &yparam, T &hx, T &hxi, T &hy,
          T &hyi, int &ier);

  static KOKKOS_INLINE_FUNCTION void
  fvbicub(Kokkos::View<int *, MemorySpace> ict, int ivec, int ivecd,
          Rank1View<T, MemorySpace> fval, const int &i, const int &j,
          const T &xparam, const T &yparam, const T &hx,
          const T &hxi, const T &hy, const T &hyi,
          Rank3View<T, MemorySpace> f);

  static KOKKOS_INLINE_FUNCTION void
  evbicub(T xget, T yget, Kokkos::View<int *, MemorySpace> ict,
          Rank1View<T, MemorySpace> fval, // output (size depends on ict)
          Rank1View<T, MemorySpace> x, const int &nx,
          Rank1View<T, MemorySpace> y, const int &ny,
          Rank3View<T, MemorySpace> f, int &ier);
  void evaluate(Kokkos::View<int *, MemorySpace> iselect,
                        Rank1View<T, MemorySpace> xvec,
                        Rank1View<T, MemorySpace> yvec,
                        Rank2View<T, MemorySpace> fval);
};

template <typename T, typename MemorySpace>
ExplicitBiCubicSplineInterpolator<T, MemorySpace>::
    ExplicitBiCubicSplineInterpolator(
        Rank1View<T, MemorySpace> x,              // size: inx
        int inx, Rank1View<T, MemorySpace> th,    // size: inth
        int inth, Rank4View<T, MemorySpace> fspl, // [4, 4, inx, inth]
        int ibcxmin,
        Rank1View<T, MemorySpace>
            bcxmin, // size: inth (used if ibcxmin = 1 or 2)
        int ibcxmax,
        Rank1View<T, MemorySpace>
            bcxmax, // size: inth (used if ibcxmax = 1 or 2)
        int ibcthmin,
        Rank1View<T, MemorySpace>
            bcthmin, // size: inx (used if ibcthmin = 1 or 2)
        int ibcthmax,
        Rank1View<T, MemorySpace>
            bcthmax,                 // size: inx (used if ibcthmax = 1 or 2)
        Rank1View<T, MemorySpace> wk // size: nwk
    ) {
  PCMS_ALWAYS_ASSERT(inx >= 2);
  PCMS_ALWAYS_ASSERT(inth >= 2);

  // Check boundary condition values
  ibc_ck(ibcxmin, "bcspline", "xmin", -1, 7);
  if (ibcxmin >= 0)
    ibc_ck(ibcxmax, "bcspline", "xmax", 0, 7);
  ibc_ck(ibcthmin, "bcspline", "thmin", -1, 7);
  if (ibcthmin >= 0)
    ibc_ck(ibcthmax, "bcspline", "thmax", 0, 7);

  CubicSplineInterpolator<T, MemorySpace>::splinck(x, 1.0e-3);
  CubicSplineInterpolator<T, MemorySpace>::splinck(th, 1.0e-3);
  this->x_ = x;
  this->nx_ = inx;
  this->y_ = th;
  this->ny_ = inth;

  bcspline(x, inx, th, inth, fspl, ibcxmin, bcxmin, ibcxmax, bcxmax, ibcthmin,
           bcthmin, ibcthmax, bcthmax, wk);

  fspl_ = fspl;
}

template <typename T, typename MemorySpace>
void ExplicitBiCubicSplineInterpolator<T, MemorySpace>::bcspline(
    Rank1View<T, MemorySpace> x,              // size: inx
    int inx, Rank1View<T, MemorySpace> th,    // size: inth
    int inth, Rank4View<T, MemorySpace> fspl, // [4, 4, inx, inth]
    int ibcxmin,
    Rank1View<T, MemorySpace> bcxmin, // size: inth (used if ibcxmin = 1 or 2)
    int ibcxmax,
    Rank1View<T, MemorySpace> bcxmax, // size: inth (used if ibcxmax = 1 or 2)
    int ibcthmin,
    Rank1View<T, MemorySpace> bcthmin, // size: inx (used if ibcthmin = 1 or 2)
    int ibcthmax,
    Rank1View<T, MemorySpace> bcthmax, // size: inx (used if ibcthmax = 1 or 2)
    Rank1View<T, MemorySpace> wk       // size: nwk
) {
  int iflg2 = 0;
  // std::vector<int> iselect1(10, 0); // Size 10, all initialized to 0
  // std::vector<int> iselect2(10, 0); // Size 10, all initialized to 0

  auto fspl_l_x = Rank2View<T, MemorySpace>(wk.data_handle() + 4 * inx * inth,
                                            4 * inx, inth);
  auto wk_l = Rank1View<T, MemorySpace>(wk.data_handle() + 2 * 4 * inx * inth,
                                        inx * inth);
  auto fspl_l_th = Rank2View<T, MemorySpace>(wk.data_handle(), 4 * inx, inth);

  if (ibcthmin != -1) {
    if (ibcthmin == 1 || ibcthmin == 2) {
      // for (int ix = 0; ix < inx; ++ix) {
      //   if (bcthmin[ix] != 0.0) {
      //     iflg2 = 1;
      //     break;
      //   }
      // }
      Kokkos::parallel_reduce(
          "check_nonzero", Kokkos::RangePolicy<execution_space>(0, inx),
          KOKKOS_LAMBDA(const int ix, int &local_flag) {
            if (bcthmin(ix) != 0.0) {
              local_flag = 1;
            }
          },
          Kokkos::Max<int>(iflg2));
    }
    if (ibcthmax == 1 || ibcthmax == 2) {
      // for (int ix = 0; ix < inx; ++ix) {
      //   if (bcthmax[ix] != 0.0) {
      //     iflg2 = 1;
      //     break;
      //   }
      // }
      Kokkos::parallel_reduce(
          "check_nonzero", Kokkos::RangePolicy<execution_space>(0, inx),
          KOKKOS_LAMBDA(const int ix, int &local_flag) {
            if (bcthmax(ix) != 0.0) {
              local_flag = 1;
            }
          },
          Kokkos::Max<int>(iflg2));
    }
  }

  T xo2 = 0.5;
  T xo6 = 1.0 / 6.0;

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
  //     CubicSplineInterpolator<T, MemorySpace>::v_spline(ibcxmin, ibcxmax,
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

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, inx), [=](int ix) {
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

          CubicSplineInterpolator<T, MemorySpace>::v_spline(
              ibcxmin, ibcxmax, inx, x, fspl_x_view, wk_x_view);
        }

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, inx), [=](int ix) {
          fspl(1, 0, ix, ith) = fspl_x_view(1, ix);
          fspl(2, 0, ix, ith) = fspl_x_view(2, ix) * xo2;
          fspl(3, 0, ix, ith) = fspl_x_view(3, ix) * xo6;
        });
      });

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
  //         CubicSplineInterpolator<T, MemorySpace>::v_spline(ibcthmina,
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
              Kokkos::TeamThreadRange(team, inth),
              [=](int ith) { fspl_th_view(0, ith) = fspl(ic, 0, ix, ith); });

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

            CubicSplineInterpolator<T, MemorySpace>::v_spline(
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

  if (iflg2 == 1) {
    int iasc = 0;        // Workspace base for correction splines
    int iinc = 4 * inth; // Spacing between correction splines

    // T zhxn = x[inx - 1] - x[inx - 2];
    // T zhth = th[inth - 1] - th[inth - 2];
    T zhxn = 0.0;
    T zhth = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, 1),
        KOKKOS_LAMBDA(const int ix, T &zhxn) {
          zhxn = x[inx - 1] - x[inx - 2];
        },
        zhxn);
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, 1),
        KOKKOS_LAMBDA(const int ith, T &zhth) {
          zhth = th[inth - 1] - th[inth - 2];
        },
        zhth);
    int jx = inx - 2;
    int jth = inth - 2;

    int iselect1_arr[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int iselect2_arr[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (ibcthmin == 1)
      iselect1_arr[2] = 1;
    if (ibcthmin == 2)
      iselect1_arr[4] = 1;
    if (ibcthmax == 1)
      iselect2_arr[2] = 1;
    if (ibcthmax == 2)
      iselect2_arr[4] = 1;
    Kokkos::View<int *, MemorySpace> iselect1("iselect1", 10);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, iselect1.size()),
        KOKKOS_LAMBDA(const int i) { iselect1(i) = iselect1_arr[i]; });
    Kokkos::View<int *, MemorySpace> iselect2("iselect2", 10);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, iselect2.size()),
        KOKKOS_LAMBDA(const int i) { iselect2(i) = iselect2_arr[i]; });

    // for (int ix = 0; ix < inx; ++ix) {
    //     T zdiff1 = 0.0, zdiff2 = 0.0;

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

    //     auto zcur = Rank2View<T, MemorySpace>(zcur_vec.data(), 1,
    //     3); if (ibcthmax == 1) {
    //         if (ix < inx - 1) {
    //             zcur(0, 0) = fspl(0, 1, ix, jth) + zhth * (2.0 * fspl(0, 2,
    //             ix, jth) + zhth * 3.0 * fspl(0, 3, ix, jth));
    //         } else {
    //             // TODO: check if this is correct
    //             bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, x, inx,
    //             th, inth, fspl, ier); if (ier != 0) return;
    //         }
    //         zdiff2 = bcthmax[ix] - zcur(0, 0);
    //     } else if (ibcthmax == 2) {
    //         if (ix < inx - 1) {
    //             zcur(0, 0) = 2.0 * fspl(0, 2, ix, jth) + 6.0 * zhth * fspl(0,
    //             3, ix, jth);
    //         } else {
    //             // TODO: check if this is correct
    //             bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, x, inx,
    //             th, inth, fspl, ier); if (ier != 0) return;
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

    //     auto fspl_s_th = Rank2View<T,
    //     MemorySpace>(fspl_l_th.data_handle() + iadr, 4, inth);
    //     CubicSplineInterpolator<T, MemorySpace>::v_spline(ibcthmin,
    //     ibcthmax, inth, th, fspl_s_th, wk_th);

    // }

    auto zcur_shared =
        Rank1View<T, MemorySpace>(wk.data_handle() + 9 * inx * inth, inx * 3);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, inx),
        KOKKOS_LAMBDA(const int ix) {
          int ier = 0;
          auto zcur =
              Rank1View<T, MemorySpace>(zcur_shared.data_handle() + ix * 3, 3);
          T zdiff1 = 0.0, zdiff2 = 0.0;
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
            zcur[0] = (ix < inx - 1)
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
              bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, x, inx, th,
                       inth, fspl, ier);
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
              bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, x, inx, th,
                       inth, fspl, ier);
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
          CubicSplineInterpolator<T, MemorySpace>::v_spline(
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

    //         CubicSplineInterpolator<T, MemorySpace>::v_spline(ibcxmin,
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

              CubicSplineInterpolator<T, MemorySpace>::v_spline(
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
void ExplicitBiCubicSplineInterpolator<T, MemorySpace>::evaluate(
    Kokkos::View<int *, MemorySpace> iselect, Rank1View<T, MemorySpace> xvec,
    Rank1View<T, MemorySpace> yvec, Rank2View<T, MemorySpace> fval) {
  PCMS_ALWAYS_ASSERT(xvec.extent(0) == yvec.extent(0) &&
                     xvec.extent(0) == fval.extent(0));

  int ivd = fval.extent(1);
  // for (size_t i = 0; i < xvec.extent(0); ++i) {
  //   int ier = 0;
  //   auto fval_view = Rank1View<T, MemorySpace>(
  //       fval.data_handle() + i * ivd, ivd);
  //   BiCubicSplineInterpolator<T, MemorySpace>::bcspeval(xvec(i), yvec(i),
  //   iselect, fval_view, x_, nx_, y_, ny_, fspl_, ier);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(0, xvec.extent(0)),
      KOKKOS_CLASS_LAMBDA(const int i) {
        int ier = 0;
        auto fval_view =
            Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd, ivd);
        bcspeval(xvec(i), yvec(i), iselect, fval_view, this->x_, this->nx_,
                 this->y_, this->ny_, this->fspl_, ier);
      });
}

template <typename T, typename MemorySpace>
void CompactBiCubicSplineInterpolator<T, MemorySpace>::evaluate(
    Kokkos::View<int *, MemorySpace> iselect, Rank1View<T, MemorySpace> xvec,
    Rank1View<T, MemorySpace> yvec, Rank2View<T, MemorySpace> fval) {

  PCMS_ALWAYS_ASSERT(xvec.extent(0) == yvec.extent(0) &&
                     xvec.extent(0) == fval.extent(0));
  int ivd = fval.extent(1);
  // for (size_t i = 0; i < xvec.extent(0); ++i) {
  //   int ier = 0;
  //   auto fval_view = Rank1View<T, MemorySpace>(
  //       fval.data_handle() + i * ivd, ivd);
  //   BiCubicSplineInterpolator<T, MemorySpace>::evbicub(
  //       xvec(i), yvec(i), iselect, fval_view,  x_, nx_, y_, ny_, f_, ier);
  // }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(0, xvec.extent(0)),
      KOKKOS_CLASS_LAMBDA(const int i) {
        int ier = 0;
        auto fval_view =
            Rank1View<T, MemorySpace>(fval.data_handle() + i * ivd, ivd);
        evbicub(xvec(i), yvec(i), iselect, fval_view, this->x_, this->nx_,
                this->y_, this->ny_, this->f_, ier);
      });
}

template <typename T, typename MemorySpace>
KOKKOS_INLINE_FUNCTION void
ExplicitBiCubicSplineInterpolator<T, MemorySpace>::bcspeval(
    T xget, T yget, Kokkos::View<int *, MemorySpace> iselect,
    Rank1View<T, MemorySpace> fval, Rank1View<T, MemorySpace> x, const int &nx,
    Rank1View<T, MemorySpace> y, const int &ny, Rank4View<T, MemorySpace> fspl,
    int &ier) {
  int i = 0;
  int j = 0;
  T dx = 0.0;
  T dy = 0.0;

  // Range finding
  bcspevxy(xget, yget, x, nx, y, ny, i, j, dx, dy, ier);
  if (ier != 0)
    return;

  // Evaluate spline function
  bcspevfn(iselect, fval, i, j, dx, dy, fspl);
}

template <typename T, typename MemorySpace>
KOKKOS_INLINE_FUNCTION void
ExplicitBiCubicSplineInterpolator<T, MemorySpace>::bcspevxy(
    T xget, T yget, Rank1View<T, MemorySpace> x, const int &nx,
    Rank1View<T, MemorySpace> y, const int &ny, int &i, int &j, T &dx,
    T &dy, int &ier) {
  int nxm = nx - 1;
  int nym = ny - 1;
  int ii, jj;

  ier = 0;
  T zxget = xget;
  T zyget = yget;

  CubicSplineInterpolator<T, MemorySpace>::range_check(
      xget, zxget, x, nx, ier);
  CubicSplineInterpolator<T, MemorySpace>::range_check(
      yget, zyget, y, ny, ier);
  

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
KOKKOS_INLINE_FUNCTION void
ExplicitBiCubicSplineInterpolator<T, MemorySpace>::bcspevfn(
    Kokkos::View<int *, MemorySpace>
        ict, // Selector array for which derivatives to compute
    Rank1View<T, MemorySpace> fval, // Output array: size [ivd, *] (flattened)
    const int &i,                   // Grid cell indices in x direction
    const int &j,                   // Grid cell indices in y direction
    const T &dx,               // x displacements within cells
    const T &dy,               // y displacements within cells
    Rank4View<T, MemorySpace> fspl  // Spline coefficients
) {
  int iaval = 0; // Index for fval
  if (ict[0] <= 2) {
    if ((ict[0] > 0) || (ict[0] == -1)) {
      // Evaluate f
      iaval++;
      fval(iaval - 1) =
          fspl(0, 0, i, j) +
          dy * (fspl(0, 1, i, j) +
                dy * (fspl(0, 2, i, j) + dy * fspl(0, 3, i, j))) +
          dx * (fspl(1, 0, i, j) +
                dy * (fspl(1, 1, i, j) +
                      dy * (fspl(1, 2, i, j) + dy * fspl(1, 3, i, j))) +
                dx * (fspl(2, 0, i, j) +
                      dy * (fspl(2, 1, i, j) +
                            dy * (fspl(2, 2, i, j) + dy * fspl(2, 3, i, j))) +
                      dx * (fspl(3, 0, i, j) +
                            dy * (fspl(3, 1, i, j) +
                                  dy * (fspl(3, 2, i, j) +
                                        dy * fspl(3, 3, i, j))))));
    }

    if ((ict[1] > 0) && (ict[0] != -1)) {
      // Evaluate df/dx
      iaval++;
      fval(iaval - 1) =
          fspl(1, 0, i, j) +
          dy * (fspl(1, 1, i, j) +
                dy * (fspl(1, 2, i, j) + dy * fspl(1, 3, i, j))) +
          2.0 * dx *
              (fspl(2, 0, i, j) +
               dy * (fspl(2, 1, i, j) +
                     dy * (fspl(2, 2, i, j) + dy * fspl(2, 3, i, j))) +
               1.5 * dx *
                   (fspl(3, 0, i, j) +
                    dy * (fspl(3, 1, i, j) +
                          dy * (fspl(3, 2, i, j) + dy * fspl(3, 3, i, j)))));
    }

    if ((ict[2] > 0) && (ict[0] != -1)) {
      // Evaluate df/dy
      iaval++;
      fval(iaval - 1) =
          fspl(0, 1, i, j) +
          dy * (2.0 * fspl(0, 2, i, j) + dy * 3.0 * fspl(0, 3, i, j)) +
          dx * (fspl(1, 1, i, j) +
                dy * (2.0 * fspl(1, 2, i, j) + dy * 3.0 * fspl(1, 3, i, j)) +
                dx * (fspl(2, 1, i, j) +
                      dy * (2.0 * fspl(2, 2, i, j) +
                            dy * 3.0 * fspl(2, 3, i, j)) +
                      dx * (fspl(3, 1, i, j) +
                            dy * (2.0 * fspl(3, 2, i, j) +
                                  dy * 3.0 * fspl(3, 3, i, j)))));
    }

    if ((ict[3] > 0) || (ict[0] == -1)) {
      // Evaluate d2f/dx2
      iaval++;
      fval(iaval - 1) =
          2.0 * (fspl(2, 0, i, j) +
                 dy * (fspl(2, 1, i, j) +
                       dy * (fspl(2, 2, i, j) + dy * fspl(2, 3, i, j)))) +
          6.0 * dx *
              (fspl(3, 0, i, j) +
               dy * (fspl(3, 1, i, j) +
                     dy * (fspl(3, 2, i, j) + dy * fspl(3, 3, i, j))));
    }

    if ((ict[4] > 0) || (ict[0] == -1)) {
      // Evaluate d2f/dy2
      iaval++;
      fval(iaval - 1) =
          2.0 * fspl(0, 2, i, j) + 6.0 * dy * fspl(0, 3, i, j) +
          dx * (2.0 * fspl(1, 2, i, j) + 6.0 * dy * fspl(1, 3, i, j) +
                dx * (2.0 * fspl(2, 2, i, j) + 6.0 * dy * fspl(2, 3, i, j) +
                      dx * (2.0 * fspl(3, 2, i, j) +
                            6.0 * dy * fspl(3, 3, i, j))));
    }

    if ((ict[5] > 0) && (ict[0] != -1)) {
      // Evaluate d2f/dxdy
      iaval++;
      fval(iaval - 1) =
          fspl(1, 1, i, j) +
          dy * (2.0 * fspl(1, 2, i, j) + dy * 3.0 * fspl(1, 3, i, j)) +
          2.0 * dx *
              (fspl(2, 1, i, j) +
               dy * (2.0 * fspl(2, 2, i, j) + dy * 3.0 * fspl(2, 3, i, j)) +
               1.5 * dx *
                   (fspl(3, 1, i, j) + dy * (2.0 * fspl(3, 2, i, j) +
                                             dy * 3.0 * fspl(3, 3, i, j))));
    }

    if (ict[0] == -1) {
      // Evaluate d4f/dx2dy2
      iaval++;
      fval(iaval - 1) =
          4.0 * fspl(2, 2, i, j) + 12.0 * dy * fspl(2, 3, i, j) +
          dx * (12.0 * fspl(3, 2, i, j) + 36.0 * dy * fspl(3, 3, i, j));
    }
  } else if (ict[0] == 3) {
    if (ict[1] == 1) {
      // d³f/dx³ (not continuous)
      iaval++;
      fval(iaval - 1) =
          6.0 * (fspl(3, 0, i, j) +
                 dy * (fspl(3, 1, i, j) +
                       dy * (fspl(3, 2, i, j) + dy * fspl(3, 3, i, j))));
    }

    if (ict[2] == 1) {
      // d³f/dx²dy
      iaval++;
      fval(iaval - 1) =
          2.0 * (fspl(2, 1, i, j) +
                 dy * (2.0 * fspl(2, 2, i, j) + dy * 3.0 * fspl(2, 3, i, j))) +
          6.0 * dx *
              (fspl(3, 1, i, j) +
               dy * (2.0 * fspl(3, 2, i, j) + dy * 3.0 * fspl(3, 3, i, j)));
    }

    if (ict[3] == 1) {
      // d³f/dxdy²
      iaval++;
      fval(iaval - 1) =
          2.0 * fspl(1, 2, i, j) + 6.0 * dy * fspl(1, 3, i, j) +
          2.0 * dx *
              (2.0 * fspl(2, 2, i, j) + 6.0 * dy * fspl(2, 3, i, j) +
               1.5 * dx *
                   (2.0 * fspl(3, 2, i, j) + 6.0 * dy * fspl(3, 3, i, j)));
    }

    if (ict[4] == 1) {
      // d³f/dy³ (not continuous)
      iaval++;
      fval(iaval - 1) =
          6.0 * (fspl(0, 3, i, j) +
                 dx * (fspl(1, 3, i, j) +
                       dx * (fspl(2, 3, i, j) + dx * fspl(3, 3, i, j))));
    }

  } else if (ict[0] == 4) {
    if (ict[1] == 1) {
      // d⁴f/dx³dy
      iaval++;
      fval(iaval - 1) =
          6.0 * (fspl(3, 1, i, j) +
                 dy * 2.0 * (fspl(3, 2, i, j) + dy * 1.5 * fspl(3, 3, i, j)));
    }

    if (ict[2] == 1) {
      // d⁴f/dx²dy²
      iaval++;
      fval(iaval - 1) =
          4.0 * fspl(2, 2, i, j) + 12.0 * dy * fspl(2, 3, i, j) +
          dx * (12.0 * fspl(3, 2, i, j) + 36.0 * dy * fspl(3, 3, i, j));
    }

    if (ict[3] == 1) {
      // d⁴f/dxdy³ (not continuous)
      iaval++;
      fval(iaval - 1) =
          6.0 * (fspl(1, 3, i, j) +
                 2.0 * dx * (fspl(2, 3, i, j) + 1.5 * dx * fspl(3, 3, i, j)));
    }

  } else if (ict[0] == 5) {
    if (ict[1] == 1) {
      // d⁵f/dx³dy² (not continuous)
      iaval++;
      fval(iaval - 1) = 12.0 * (fspl(3, 2, i, j) + dy * 3.0 * fspl(3, 3, i, j));
    }

    if (ict[2] == 1) {
      // d⁵f/dx²dy³ (not continuous)
      iaval++;
      fval(iaval - 1) = 12.0 * (fspl(2, 3, i, j) + dx * 3.0 * fspl(3, 3, i, j));
    }

  } else if (ict[0] == 6) {
    // d⁶f/dx³dy³ (not continuous)
    iaval++;
    fval(iaval - 1) = 36.0 * fspl(3, 3, i, j);
  }
} // end bcspevfn

template <typename T, typename MemorySpace>
CompactBiCubicSplineInterpolator<T, MemorySpace>::
    CompactBiCubicSplineInterpolator(
        Rank1View<T, MemorySpace> x, int nx, Rank1View<T, MemorySpace> y,
        int ny, Rank3View<T, MemorySpace> f, int ibcxmin,
        Rank1View<T, MemorySpace> bcxmin, int ibcxmax,
        Rank1View<T, MemorySpace> bcxmax, int ibcymin,
        Rank1View<T, MemorySpace> bcymin, int ibcymax,
        Rank1View<T, MemorySpace> bcymax, Rank1View<T, MemorySpace> wk) {
  // Check bc validity (implement ibc_ck separately)
  ibc_ck(ibcxmin, "bcspline", "xmin", -1, 7);
  if (ibcxmin >= 0)
    ibc_ck(ibcxmax, "bcspline", "xmax", 0, 7);
  ibc_ck(ibcymin, "bcspline", "ymin", -1, 7);
  if (ibcymin >= 0)
    ibc_ck(ibcymax, "bcspline", "ymax", 0, 7);

  CubicSplineInterpolator<T, MemorySpace>::splinck(x, 1.0e-3);
  CubicSplineInterpolator<T, MemorySpace>::splinck(y, 1.0e-3);
  this->x_ = x;   // Store the x-coordinates
  this->y_ = y;   // Store the y-coordinates
  this->nx_ = nx; // Store the number of x-coordinates
  this->ny_ = ny; // Store the number of y-coordinates
  mkbicub(x, nx, y, ny, f, ibcxmin, bcxmin, ibcxmax, bcxmax, ibcymin, bcymin,
          ibcymax, bcymax, wk);
  this->f_ = f; // Store the spline coefficients
}

template <typename T, typename MemorySpace>
void CompactBiCubicSplineInterpolator<T, MemorySpace>::mkbicub(
    Rank1View<T, MemorySpace> x, int nx, Rank1View<T, MemorySpace> y, int ny,
    Rank3View<T, MemorySpace> f, int ibcxmin, Rank1View<T, MemorySpace> bcxmin,
    int ibcxmax, Rank1View<T, MemorySpace> bcxmax, int ibcymin,
    Rank1View<T, MemorySpace> bcymin, int ibcymax,
    Rank1View<T, MemorySpace> bcymax, Rank1View<T, MemorySpace> wk) {
  int iflg2 = 0;

  // Check if inhomogeneous y-boundary conditions exist
  if (ibcymin != -1) {
    if ((ibcymin == 1 || ibcymin == 2)) {
      // for (int ix = 0; ix < nx; ++ix) {
      //   if (bcymin[ix] != 0.0)
      //     iflg2 = 1;
      // }
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<execution_space>(0, nx),
          KOKKOS_LAMBDA(const int ix, int &local_iflg2) {
            if (bcymin[ix] != 0.0) {
              local_iflg2 = 1;
            }
          },
          Kokkos::Max<int>(iflg2));
    }
    if ((ibcymax == 1 || ibcymax == 2)) {
      // for (int ix = 0; ix < nx; ++ix) {
      //   if (bcymax[ix] != 0.0)
      //     iflg2 = 1;
      // }
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<execution_space>(0, nx),
          KOKKOS_LAMBDA(const int ix, int &local_iflg2) {
            if (bcymax[ix] != 0.0) {
              local_iflg2 = 1;
            }
          },
          Kokkos::Max<int>(iflg2));
    }
  }

  auto fspl_l_x = Rank2View<T, MemorySpace>(wk.data_handle(), 2 * ny, nx);
  auto wk_l =
      Rank1View<T, MemorySpace>(wk.data_handle() + 2 * ny * nx, nx * ny);
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
        T zbcmin = 0.0, zbcmax = 0.0;

        auto fwk_x_view = Rank2View<T, MemorySpace>(
            fspl_l_x.data_handle() + 2 * nx * iy, 2, nx);
        auto wk_x_view =
            Rank1View<T, MemorySpace>(wk_l.data_handle() + nx * iy, nx);
        auto fwk4_x_view = Rank2View<T, MemorySpace>(
            fwk4_l_x.data_handle() + 4 * nx * iy, 4, nx);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx),
                             [=](int ix) { fwk_x_view(0, ix) = f(0, ix, iy); });

        if (team.team_rank() == 0) {
          if (ibcxmin == 1 || ibcxmin == 2)
            zbcmin = bcxmin[iy];
          if (ibcxmax == 1 || ibcxmax == 2)
            zbcmax = bcxmax[iy];

          CompactCubicSplineInterpolator<T, MemorySpace>::mkspline(
              x, nx, fwk_x_view, fwk4_x_view, ibcxmin, zbcmin, ibcxmax, zbcmax,
              wk_x_view);
        }

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx),
                             [=](int ix) { f(1, ix, iy) = fwk_x_view(1, ix); });
      });

  // T zbcmin = 0.0, zbcmax = 0.0;
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
        int ibcmin = ibcymin, ibcmax = ibcymax;

        auto fwk_y_view = Rank2View<T, MemorySpace>(
            fspl_l_x.data_handle() + 2 * ny * ix, 2, ny);
        auto wk_y_view =
            Rank1View<T, MemorySpace>(wk_l.data_handle() + ny * ix, ny);
        auto fwk4_y_view = Rank2View<T, MemorySpace>(
            fwk4_l_x.data_handle() + 4 * ny * ix, 4, ny);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny),
                             [=](int iy) { fwk_y_view(0, iy) = f(0, ix, iy); });

        if (team.team_rank() == 0) {
          if (iflg2 == 1) {
            if (ibcymin == 1 || ibcymin == 2)
              ibcmin = 0;
            if (ibcymax == 1 || ibcymax == 2)
              ibcmax = 0;
          }

          CompactCubicSplineInterpolator<T, MemorySpace>::mkspline(
              y, ny, fwk_y_view, fwk4_y_view, ibcmin, 0.0, ibcmax, 0.0,
              wk_y_view);
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny),
                             [=](int iy) { f(2, ix, iy) = fwk_y_view(1, iy); });
      });

  // int zbcmin = 0.0;
  // int zbcmax = 0.0;
  // T ibcmin = ibcymin;
  // T ibcmax = ibcymax;
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
        int ibcmin = ibcymin, ibcmax = ibcymax;

        auto fwk_y_view = Rank2View<T, MemorySpace>(
            fspl_l_x.data_handle() + 2 * ny * ix, 2, ny);
        auto wk_y_view =
            Rank1View<T, MemorySpace>(wk_l.data_handle() + ny * ix, ny);
        auto fwk4_y_view = Rank2View<T, MemorySpace>(
            fwk4_l_x.data_handle() + 4 * ny * ix, 4, ny);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny),
                             [=](int iy) { fwk_y_view(0, iy) = f(1, ix, iy); });

        if (team.team_rank() == 0) {
          if (iflg2 == 1) {
            if (ibcymin == 1 || ibcymin == 2)
              ibcmin = 0;
            if (ibcymax == 1 || ibcymax == 2)
              ibcmax = 0;
          }

          CompactCubicSplineInterpolator<T, MemorySpace>::mkspline(
              y, ny, fwk_y_view, fwk4_y_view, ibcmin, 0.0, ibcmax, 0.0,
              wk_y_view);
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny),
                             [=](int iy) { f(3, ix, iy) = fwk_y_view(1, iy); });
      });

  int zbcmin = 0.0;
  int zbcmax = 0.0;

  // Correct for inhomogeneous boundary conditions if needed
  if (iflg2 == 1) {
    auto fcorr =
        Rank3View<T, MemorySpace>(wk.data_handle() + 7 * ny * nx, 2, nx, ny);

    // T zdiff1 = 0.0, zdiff2 = 0.0;
    // auto fwk_y = Rank2View<T, MemorySpace>(
    //     fspl_l_x.data_handle(), 2, ny);
    // auto wk_y = Rank1View<T, MemorySpace>(
    //     wk.data_handle(), ny);
    // auto fwk4_y = Rank2View<T, MemorySpace>(
    //     fwk4_l_x.data_handle(), 4, ny);
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
    //     fwk4_y, ibcymin, zdiff1, ibcymax, zdiff2, wk_y);
    //     for (int iy = 0; iy < ny; ++iy)
    //         fcorr(0, ix, iy) = fwk_y(1, iy);
    // }

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<execution_space>(nx, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type &team) {
          const int ix = team.league_rank();
          T zdiff1 = 0.0, zdiff2 = 0.0;

          auto fwk_y_view = Rank2View<T, MemorySpace>(
              fspl_l_x.data_handle() + 2 * ny * ix, 2, ny);
          auto wk_y_view =
              Rank1View<T, MemorySpace>(wk_l.data_handle() + ny * ix, ny);
          auto fwk4_y_view = Rank2View<T, MemorySpace>(
              fwk4_l_x.data_handle() + 4 * ny * ix, 4, ny);

          if (ibcymin == 1) {
            zdiff1 = bcymin[ix] -
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

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny),
                               [=](int iy) { fwk_y_view(0, iy) = 0.0; });
          if (team.team_rank() == 0) {
            CompactCubicSplineInterpolator<T, MemorySpace>::mkspline(
                y, ny, fwk_y_view, fwk4_y_view, ibcymin, zdiff1, ibcymax,
                zdiff2, wk_y_view);
          }
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny), [=](int iy) {
            fcorr(0, ix, iy) = fwk_y_view(1, iy);
          });
        });

    // zbcmin=0;
    // zbcmax=0;
    // auto fwk_x = Rank2View<T, MemorySpace>(fspl_l_x.data_handle(), 2, nx);
    // auto fwk4_x = Rank2View<T, MemorySpace>(fwk4_l_x.data_handle(), 4, nx);
    // auto wk_x = Rank1View<T, MemorySpace>(wk_l.data_handle(), nx);
    // // Compute fxy
    // for (int iy = 0; iy < ny; ++iy) {
    //     for (int ix = 0; ix < nx; ++ix)
    //         fwk_x(0, ix) = fcorr(0, ix, iy);
    //     CubicSplineInterpolator<T, MemorySpace>::mkspline(x, nx, fwk_x,
    //     fwk4_x, ibcxmin, 0.0, ibcxmax, 0.0, wk_x);
    //     for (int ix = 0; ix < nx; ++ix)
    //         fcorr(1, ix, iy) = fwk_x(1, ix);
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

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx), [=](int ix) {
            fwk_x_view(0, ix) = fcorr(0, ix, iy);
          });

          if (team.team_rank() == 0) {
            CompactCubicSplineInterpolator<T, MemorySpace>::mkspline(
                x, nx, fwk_x_view, fwk4_x_view, ibcxmin, zbcmin, ibcxmax,
                zbcmax, wk_x_view);
          }

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx), [=](int ix) {
            fcorr(1, ix, iy) = fwk_x_view(1, ix);
          });
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
KOKKOS_INLINE_FUNCTION void CompactBiCubicSplineInterpolator<T, MemorySpace>::herm2xy(
    T xget, T yget, Rank1View<T, MemorySpace> x, const int &nx,
    Rank1View<T, MemorySpace> y, const int &ny, int &i, int &j, T &xparam,
    T &yparam, T &hx, T &hxi, T &hy, T &hyi,
    int &ier) {
  ier = 0;

  T zxget = xget;
  T zyget = yget;

  CubicSplineInterpolator<T, MemorySpace>::range_check(
      xget, zxget, x, nx, ier);
  CubicSplineInterpolator<T, MemorySpace>::range_check(
      yget, zyget, y, ny, ier);

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
KOKKOS_INLINE_FUNCTION void CompactBiCubicSplineInterpolator<T, MemorySpace>::fvbicub(
    Kokkos::View<int *, MemorySpace> ict, int ivec, int ivecd,
    Rank1View<T, MemorySpace> fval, const int &i, const int &j,
    const T &xparam, const T &yparam, const T &hx,
    const T &hxi, const T &hy, const T &hyi,
    Rank3View<T, MemorySpace> f) {
  constexpr T sixth = 1.0 / 6.0;
  const T z36th = sixth * sixth;
  int iadr = 0;

  /* ------------------------------------------------------------------
   * ict[0] = 1 or 2  →   f, df/dx, df/dy, d²f/dx², d²f/dy², d²f/dxdy
   * ----------------------------------------------------------------*/
  if (ict[0] <= 2) {
    /********** f (function value) **********/
    if (ict[0] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T xp2 = xp * xp, xpi2 = xpi * xpi;
      T cx = xp * (xp2 - 1.0);
      T cxi = xpi * (xpi2 - 1.0);
      T hx2 = hx * hx;

      T yp = yparam, ypi = 1.0 - yp;
      T yp2 = yp * yp, ypi2 = ypi * ypi;
      T cy = yp * (yp2 - 1.0);
      T cyi = ypi * (ypi2 - 1.0);
      T hy2 = hy * hy;

      T sum = xpi * (ypi * f(0, i, j) + yp * f(0, i, j + 1)) +
                   xp * (ypi * f(0, i + 1, j) + yp * f(0, i + 1, j + 1));
      sum += sixth * hx2 *
             (cxi * (ypi * f(1, i, j) + yp * f(1, i, j + 1)) +
              cx * (ypi * f(1, i + 1, j) + yp * f(1, i + 1, j + 1)));
      sum += sixth * hy2 *
             (xpi * (cyi * f(2, i, j) + cy * f(2, i, j + 1)) +
              xp * (cyi * f(2, i + 1, j) + cy * f(2, i + 1, j + 1)));
      sum += z36th * hx2 * hy2 *
             (cxi * (cyi * f(3, i, j) + cy * f(3, i, j + 1)) +
              cx * (cyi * f(3, i + 1, j) + cy * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** df/dx **********/
    if (ict[1] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T xp2 = xp * xp, xpi2 = xpi * xpi;
      T cxd = 3.0 * xp2 - 1.0;
      T cxdi = -3.0 * xpi2 + 1.0;
      T yp = yparam, ypi = 1.0 - yp;
      T yp2 = yp * yp, ypi2 = ypi * ypi;
      T cy = yp * (yp2 - 1.0);
      T cyi = ypi * (ypi2 - 1.0);
      T hy2 = hy * hy;

      T sum = hxi * (-(ypi * f(0, i, j) + yp * f(0, i, j + 1)) +
                          (ypi * f(0, i + 1, j) + yp * f(0, i + 1, j + 1)));
      sum += sixth * hx *
             (cxdi * (ypi * f(1, i, j) + yp * f(1, i, j + 1)) +
              cxd * (ypi * f(1, i + 1, j) + yp * f(1, i + 1, j + 1)));
      sum += sixth * hxi * hy2 *
             (-(cyi * f(2, i, j) + cy * f(2, i, j + 1)) +
              (cyi * f(2, i + 1, j) + cy * f(2, i + 1, j + 1)));
      sum += z36th * hx * hy2 *
             (cxdi * (cyi * f(3, i, j) + cy * f(3, i, j + 1)) +
              cxd * (cyi * f(3, i + 1, j) + cy * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** df/dy **********/
    if (ict[2] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T xp2 = xp * xp, xpi2 = xpi * xpi;
      T cx = xp * (xp2 - 1.0);
      T cxi = xpi * (xpi2 - 1.0);
      T hx2 = hx * hx;
      T yp = yparam, ypi = 1.0 - yp;
      T yp2 = yp * yp, ypi2 = ypi * ypi;
      T cyd = 3.0 * yp2 - 1.0;
      T cydi = -3.0 * ypi2 + 1.0;

      T sum = hyi * (xpi * (-f(0, i, j) + f(0, i, j + 1)) +
                          xp * (-f(0, i + 1, j) + f(0, i + 1, j + 1)));
      sum += sixth * hx2 * hyi *
             (cxi * (-f(1, i, j) + f(1, i, j + 1)) +
              cx * (-f(1, i + 1, j) + f(1, i + 1, j + 1)));
      sum += sixth * hy *
             (xpi * (cydi * f(2, i, j) + cyd * f(2, i, j + 1)) +
              xp * (cydi * f(2, i + 1, j) + cyd * f(2, i + 1, j + 1)));
      sum += z36th * hx2 * hy *
             (cxi * (cydi * f(3, i, j) + cyd * f(3, i, j + 1)) +
              cx * (cydi * f(3, i + 1, j) + cyd * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d²f/dx² **********/
    if (ict[3] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T yp = yparam, ypi = 1.0 - yp;
      T yp2 = yp * yp, ypi2 = ypi * ypi;
      T cy = yp * (yp2 - 1.0);
      T cyi = ypi * (ypi2 - 1.0);
      T hy2 = hy * hy;

      T sum = xpi * (ypi * f(1, i, j) + yp * f(1, i, j + 1)) +
                   xp * (ypi * f(1, i + 1, j) + yp * f(1, i + 1, j + 1));
      sum += sixth * hy2 *
             (xpi * (cyi * f(3, i, j) + cy * f(3, i, j + 1)) +
              xp * (cyi * f(3, i + 1, j) + cy * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d²f/dy² **********/
    if (ict[4] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T xp2 = xp * xp, xpi2 = xpi * xpi;
      T cx = xp * (xp2 - 1.0);
      T cxi = xpi * (xpi2 - 1.0);
      T hx2 = hx * hx;
      T yp = yparam, ypi = 1.0 - yp;

      T sum = xpi * (ypi * f(2, i, j) + yp * f(2, i, j + 1)) +
                   xp * (ypi * f(2, i + 1, j) + yp * f(2, i + 1, j + 1));
      sum += sixth * hx2 *
             (cxi * (ypi * f(3, i, j) + yp * f(3, i, j + 1)) +
              cx * (ypi * f(3, i + 1, j) + yp * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d²f/dxdy **********/
    if (ict[5] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T xp2 = xp * xp, xpi2 = xpi * xpi;
      T cxd = 3.0 * xp2 - 1.0;
      T cxdi = -3.0 * xpi2 + 1.0;
      T yp = yparam, ypi = 1.0 - yp;
      T yp2 = yp * yp, ypi2 = ypi * ypi;
      T cyd = 3.0 * yp2 - 1.0;
      T cydi = -3.0 * ypi2 + 1.0;

      T sum =
          hxi * hyi *
          (f(0, i, j) - f(0, i, j + 1) - f(0, i + 1, j) + f(0, i + 1, j + 1));
      sum += sixth * hx * hyi *
             (cxdi * (-f(1, i, j) + f(1, i, j + 1)) +
              cxd * (-f(1, i + 1, j) + f(1, i + 1, j + 1)));
      sum += sixth * hxi * hy *
             (-(cydi * f(2, i, j) + cyd * f(2, i, j + 1)) +
              (cydi * f(2, i + 1, j) + cyd * f(2, i + 1, j + 1)));
      sum += z36th * hx * hy *
             (cxdi * (cydi * f(3, i, j) + cyd * f(3, i, j + 1)) +
              cxd * (cydi * f(3, i + 1, j) + cyd * f(3, i + 1, j + 1)));
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
      T yp = yparam, ypi = 1.0 - yp;
      T yp2 = yp * yp, ypi2 = ypi * ypi;
      T cy = yp * (yp2 - 1.0);
      T cyi = ypi * (ypi2 - 1.0);
      T hy2 = hy * hy;
      T sum = hxi * (-(ypi * f(1, i, j) + yp * f(1, i, j + 1)) +
                          (ypi * f(1, i + 1, j) + yp * f(1, i + 1, j + 1)));
      sum += sixth * hy2 * hxi *
             (-(cyi * f(3, i, j) + cy * f(3, i, j + 1)) +
              (cyi * f(3, i + 1, j) + cy * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d³f/dx²dy **********/
    if (ict[2] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T yp = yparam, ypi = 1.0 - yp;
      T yp2 = yp * yp, ypi2 = ypi * ypi;
      T cyd = 3.0 * yp2 - 1.0;
      T cydi = -3.0 * ypi2 + 1.0;
      T sum = hyi * (xpi * (-f(1, i, j) + f(1, i, j + 1)) +
                          xp * (-f(1, i + 1, j) + f(1, i + 1, j + 1)));
      sum += sixth * hy *
             (xpi * (cydi * f(3, i, j) + cyd * f(3, i, j + 1)) +
              xp * (cydi * f(3, i + 1, j) + cyd * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d³f/dxdy² **********/
    if (ict[3] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T xp2 = xp * xp, xpi2 = xpi * xpi;
      T cxd = 3.0 * xp2 - 1.0;
      T cxdi = -3.0 * xpi2 + 1.0;
      T yp = yparam, ypi = 1.0 - yp;
      T sum = hxi * (-(ypi * f(2, i, j) + yp * f(2, i, j + 1)) +
                          (ypi * f(2, i + 1, j) + yp * f(2, i + 1, j + 1)));
      sum += sixth * hx *
             (cxdi * (ypi * f(3, i, j) + yp * f(3, i, j + 1)) +
              cxd * (ypi * f(3, i + 1, j) + yp * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d³f/dy³ **********/
    if (ict[4] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T xp2 = xp * xp, xpi2 = xpi * xpi;
      T cx = xp * (xp2 - 1.0);
      T cxi = xpi * (xpi2 - 1.0);
      T hx2 = hx * hx;
      T sum = hyi * (xpi * (-f(2, i, j) + f(2, i, j + 1)) +
                          xp * (-f(2, i + 1, j) + f(2, i + 1, j + 1)));
      sum += sixth * hx2 * hyi *
             (cxi * (-f(3, i, j) + f(3, i, j + 1)) +
              cx * (-f(3, i + 1, j) + f(3, i + 1, j + 1)));
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
      T yp = yparam, ypi = 1.0 - yp;
      T yp2 = yp * yp, ypi2 = ypi * ypi;
      T cyd = 3.0 * yp2 - 1.0;
      T cydi = -3.0 * ypi2 + 1.0;
      T sum =
          hxi * hyi *
          (f(1, i, j) - f(1, i, j + 1) - f(1, i + 1, j) + f(1, i + 1, j + 1));
      sum += sixth * hy * hxi *
             (-(cydi * f(3, i, j) + cyd * f(3, i, j + 1)) +
              (cydi * f(3, i + 1, j) + cyd * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d⁴f/dx²dy² **********/
    if (ict[2] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T yp = yparam, ypi = 1.0 - yp;
      T sum = xpi * (ypi * f(3, i, j) + yp * f(3, i, j + 1)) +
                   xp * (ypi * f(3, i + 1, j) + yp * f(3, i + 1, j + 1));
      fval(iadr - 1) = sum;
    }

    /********** d⁴f/dxdy³ **********/
    if (ict[3] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T xp2 = xp * xp, xpi2 = xpi * xpi;
      T cxd = 3.0 * xp2 - 1.0;
      T cxdi = -3.0 * xpi2 + 1.0;
      T sum =
          hxi * hyi *
          (f(2, i, j) - f(2, i, j + 1) - f(2, i + 1, j) + f(2, i + 1, j + 1));
      sum += sixth * hx * hyi *
             (cxdi * (-f(3, i, j) + f(3, i, j + 1)) +
              cxd * (-f(3, i + 1, j) + f(3, i + 1, j + 1)));
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
      T yp = yparam, ypi = 1.0 - yp;
      T sum = hxi * (-(ypi * f(3, i, j) + yp * f(3, i, j + 1)) +
                          (ypi * f(3, i + 1, j) + yp * f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }

    /********** d⁵f/dx²dy³ **********/
    if (ict[2] == 1) {
      iadr++;
      T xp = xparam, xpi = 1.0 - xp;
      T sum = hyi * (xpi * (-f(3, i, j) + f(3, i, j + 1)) +
                          xp * (-f(3, i + 1, j) + f(3, i + 1, j + 1)));
      fval(iadr - 1) = sum;
    }
  }

  /* ------------------------------------------------------------------
   * ict[0] = 6  → 6th‑order derivative (dx³dy³)
   * ----------------------------------------------------------------*/
  else if (ict[0] == 6) {
    iadr++;
    T sum =
        hxi * hyi *
        (f(3, i, j) - f(3, i, j + 1) - f(3, i + 1, j) + f(3, i + 1, j + 1));
    fval(iadr - 1) = sum;
  }
}

template <typename T, typename MemorySpace>
KOKKOS_INLINE_FUNCTION void CompactBiCubicSplineInterpolator<T, MemorySpace>::evbicub(
    T xget, T yget, Kokkos::View<int *, MemorySpace> ict,
    Rank1View<T, MemorySpace> fval, // output (size depends on ict)
    Rank1View<T, MemorySpace> x, const int &nx, Rank1View<T, MemorySpace> y,
    const int &ny, Rank3View<T, MemorySpace> f, // input (size depends on ict)
    int &ier) {
  // Local variables
  int i = 0, j = 0;
  T xparam = 0.0, yparam = 0.0;
  T hx = 0.0, hy = 0.0;
  T hxi = 0.0, hyi = 0.0;

  // Call herm2xy to locate cell and compute params
  herm2xy(xget, yget, x, nx, y, ny, i, j, xparam, yparam, hx, hxi, hy, hyi,
          ier);

  if (ier != 0)
    return;

  // Call fvbicub with scalar-vector interface
  fvbicub(ict, 1, 1, fval, i, j, xparam, yparam, hx, hxi, hy, hyi, f);
}

} // namespace pcms
#endif