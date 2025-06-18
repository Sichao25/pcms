
#ifndef MLS_RBF_OPTIONS_HPP
#define MLS_RBF_OPTIONS_HPP

#include <cmath>
#include "pcms/arrays.h"
#include "pcms/memory_spaces.h"
#include "pcms/types.h"
#include "mdspan/mdspan.hpp"
#include <Kokkos_Core.hpp>

namespace pcms
{

// Todo: remove this once target PR is merged
// Declaration of customized Kokkos types
template <typename ElementType, typename MemorySpace>
struct memory_space_accessor : public Kokkos::default_accessor<ElementType>
{
  using memory_space = MemorySpace;
};

template <int Rank, typename ElementType, typename MemorySpace>
using View =
  Kokkos::mdspan<ElementType, Kokkos::dextents<LO, Rank>, Kokkos::layout_right,
                 detail::memory_space_accessor<
                   std::remove_reference_t<ElementType>, MemorySpace>>;

template <typename ElementType, typename MemorySpace>
using Rank1View = View<1, ElementType, MemorySpace>;

template <typename ElementType, typename MemorySpace>
using Rank2View = View<2, ElementType, MemorySpace>;

template <typename T, typename MemorySpace>
class SplineInterpolator {
public:
    SplineInterpolator() = default;
};

// TODO: better error handling, use pcms existing error handling
// TODO: better documentation
// TODO: add more checks/assetions for input parameters
template <typename T, typename MemorySpace>
class CubicSplineInterpolator: public SplineInterpolator<T, MemorySpace> {
public:

    CubicSplineInterpolator() = default;

    void cspline(Rank1View<const T, MemorySpace> x, int& nx, Rank2View<T, MemorySpace> fspl, const int& ibcxmin,
        const double& bcxmin, const int& ibcxmax, const double& bcxmax,
        Rank1View<T, MemorySpace> wk, const int& iwk);

    void v_spline(const int& k_bc1, const int& k_bcn, const int& n, Rank1View<const T, MemorySpace> x,
        Rank2View<T, MemorySpace> f, Rank1View<T, MemorySpace> wk);

    void splinck(Rank1View<const T, MemorySpace>& x, const double& ztol);

    void spvec(
        const std::vector<int>& selector, // ict[0] = f, ict[1] = df/dx, ict[2] = d²f/dx²
        const int& ivec, // number of x values
        Rank1View<const T, MemorySpace> xvec, // input x vector
        const int& ivd, // output dimension >= ivec
        Rank2View<T, MemorySpace> fval, // output values: fval(ivec, up to 3)
        const int& nx, // number of x grid points
        Rank2View<T, MemorySpace> xpkg, // x package: xpkg(nx, 4)
        Rank2View<T, MemorySpace> fspl, // spline coefficients: fspl(4, nx)
        int& iwarn // warning flag
    );

    void genxpkg(int nx,
        Rank1View<const T, MemorySpace> x,
        Rank2View<T, MemorySpace> xpkg,
        int iper,
        int imsg,
        const int& itol,
        const double& ztol,
        const int& ialg
    );


    void cspevfn(const std::vector<int>& selector, const int& ivec, const int& ivd,
        Rank2View<T, MemorySpace> fval,
        std::vector<int>& iv, std::vector<double>& dxv,
        Rank2View<T, MemorySpace> f, int nx
    );

    void spgrid(
        Rank1View<const T, MemorySpace>& x_newgrid,
        const int& nx_new,
        Rank2View<T, MemorySpace>& f_new,
        const int& nx,
        Rank2View<T, MemorySpace>& xpkg,
        Rank2View<T, MemorySpace>& fspl,
        int& iwarn
    );


    void xlookup(
        const int& ivec,
        Rank1View<const T, MemorySpace> xvec,
        const int& nx,
        Rank2View<T, MemorySpace> xpkg,
        const int& imode,
        std::vector<int>& iv,
        std::vector<double>& dxn,
        std::vector<double>& hv,
        std::vector<double>& hiv,
        int& iwarn
    );

}; // end of cubic spline class


template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::cspline(Rank1View<const T, MemorySpace> x, int& nx, Rank2View<T, MemorySpace> fspl, const int& ibcxmin,
    const double& bcxmin, const int& ibcxmax, const double& bcxmax, Rank1View<T, MemorySpace> wk, const int& iwk) {
    if (x.extent(0) < 2) {
        throw std::invalid_argument("Cubic spline requires at least 2 points in each dimension.");
    }
    // TODO: check min max between -1, 0 - 7 ?

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
    v_spline(ibcxmin,ibcxmax,nx,x,fspl,wk);
    for (int i = 0; i < nx; ++i) {
        fspl(2, i) = half * fspl(2, i);
        fspl(3, i) = sixth * fspl(3, i);
    }
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::v_spline(const int& k_bc1, const int& k_bcn, const int& n, Rank1View<const T, MemorySpace> x,
    Rank2View<T, MemorySpace> f, Rank1View<T, MemorySpace> wk) {
    int i_bc1 = k_bc1;
    int i_bcn = k_bcn;
    int iord1, iord2, imin, imax;
    double a1, b1, an, bn, f0, fh, h;


    // Clip to allowed ranges
    if (i_bc1 < -1 || i_bc1 > 7) i_bc1 = 0;  // outside [-1,7] -> not-a-knot
    if (i_bcn <  0 || i_bcn > 7) i_bcn = 0;  // outside [0,7]  -> not-a-knot

    // Periodic BC handling
    if (i_bc1 == -1) i_bcn = -1;

    int i3knots = 0;
    int i3perio = 0;
    // TODO: revisit BC check
    if (n == 3) {
        i_bc1 = std::min(6, i_bc1);
        i_bcn = std::min(6, i_bcn);

        if (i_bc1 == 0) i3knots += 1;
        if (i_bcn == 0) i3knots += 1;
        if (i_bc1 == -1) i3perio = 1;
    }

    if (n == 2) {
        if (i_bc1 == -1) {
            i_bc1 = 5;
            i_bcn = 5;
        }

        if (i_bc1 == 0 || i_bc1 > 5) i_bc1 = 5;
        if (i_bcn == 0 || i_bcn > 5) i_bcn = 5;

        // LHS match
        if (i_bc1 == 1 || i_bc1 == 3 || i_bc1 == 5)
            iord1 = 1;  // first derivative match
        else
            iord1 = 2;  // second derivative match
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
        b1 = 2.0 * ((f(0, 2) - f(0, 1)) / (x[2] - x[1]) -
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
        bn = 2.0 * ((f(0, n - 1) - f(0, n - 2)) / (x[n - 1] - x[n - 2]) -
                    (f(0, n - 2) - f(0, n - 3)) / (x[n - 2] - x[n - 3])) /
                    (x[n - 1] - x[n - 3]);
    }
    f(1, n-1) = 0.0;
    f(2, n-1) = 0.0;
    f(3, n-1) = 0.0;
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

            f(2, 0) = (3 * (fh - f0) / (h * h) - (2 * a1 + an) / h) * 2;       // 2*c2
            f(3, 0) = (-2 * (fh - f0) / (h * h * h) + (a1 + an) / (h * h)) * 6; // 6*c1

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
            f(3, 0) = (bn / (4 * h) - (fh - f0) / (2 * h * h * h) + a1 / (2 * h * h)) * 6;

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
            f(3, 0) = (an / (2 * h * h) - (fh - f0) / (2 * h * h * h) - b1 / (4 * h)) * 6;

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

            double aa = a1 / (h2 * h3) + f3 / (h3 * h3 * (h3 - h2)) - f2 / (h2 * h2 * (h3 - h2));
            double bb = -a1 * (h3 * h3 - h2 * h2) / (h2 * h3 * (h3 - h2))
                        + f2 * h3 / (h2 * h2 * (h3 - h2)) - f3 * h2 / (h3 * h3 * (h3 - h2));

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

            double aa = -(b1 / 2.0) * (h3 - h2) / (h3 * h3 - h2 * h2)
                        - f2 / (h2 * (h3 * h3 - h2 * h2)) + f3 / (h3 * (h3 * h3 - h2 * h2));
            double bb = -(b1 / 2.0) * h2 * h3 * (h3 - h2) / (h3 * h3 - h2 * h2)
                        + f2 * h3 * h3 / (h2 * (h3 * h3 - h2 * h2))
                        - f3 * h2 * h2 / (h3 * (h3 * h3 - h2 * h2));

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

            double aa = an / (h2 * h3) + f3 / (h3 * h3 * (h3 - h2)) - f2 / (h2 * h2 * (h3 - h2));
            double bb = -an * (h3 * h3 - h2 * h2) / (h2 * h3 * (h3 - h2))
                        + f2 * h3 / (h2 * h2 * (h3 - h2)) - f3 * h2 / (h3 * h3 * (h3 - h2));

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

            double aa = -(bn / 2.0) * (h3 - h2) / (h3 * h3 - h2 * h2)
                        - f2 / (h2 * (h3 * h3 - h2 * h2)) + f3 / (h3 * (h3 * h3 - h2 * h2));
            double bb = -(bn / 2.0) * h2 * h3 * (h3 - h2) / (h3 * h3 - h2 * h2)
                        + f2 * h3 * h3 / (h2 * (h3 * h3 - h2 * h2))
                        - f3 * h2 * h2 / (h3 * (h3 * h3 - h2 * h2));

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

        for (int i = 1; i < n - 1; ++i) {
            f(3, i) = x[i + 1] - x[i];
            f(1, i) = 2.0 * (f(3, i - 1) + f(3, i));
            f(2, i + 1) = (f(0, i + 1) - f(0, i)) / f(3, i);
            f(2, i) = f(2, i + 1) - f(2, i);
        }

        double elem21 = f(3, 0);
        double elemnn1 = f(3, n - 2);

        // Left boundary conditions
        if (i_bc1 == -1) {
            f(1, 0) = 2.0 * (f(3, 0) + f(3, n - 2));
            f(2, 0) = (f(0, 1) - f(0, 0)) / f(3, 0) - (f(0, n - 1) - f(0, n - 2)) / f(3, n - 2);
            wk[0] = f(3, n - 2);
            for (int i = 1; i <= n - 4; ++i) wk[i] = 0.0;
            wk[n - 3] = f(3, n - 3);
            wk[n - 2] = f(3, n - 2);
        }
        else if (i_bc1 == 1 || i_bc1 == 3 || i_bc1 == 5) {
            f(1, 0) = 2.0 * f(3, 0);
            f(2, 0) = (f(0, 1) - f(0, 0)) / f(3, 0) - a1;
        }
        else if (i_bc1 == 2 || i_bc1 == 4 || i_bc1 == 6) {
            f(1, 0) = 2.0 * f(3, 0);
            f(2, 0) = f(3, 0) * b1 / 3.0;
            f(3, 0) = 0.0;
        }
        else if (i_bc1 == 7) {
            f(1, 0) = -f(3, 0);
            f(2, 0) = f(2, 2) / (x[3] - x[1]) - f(2, 1) / (x[2] - x[0]);
            f(2, 0) *= f(3, 0) * f(3, 0) / (x[3] - x[0]);
        }
        else {
            imin = 1;
            f(1, 1) = f(3, 0) + 2.0 * f(3, 1);
            f(2, 1) = f(2, 1) * f(3, 1) / (f(3, 0) + f(3, 1));
        }

        // Right boundary conditions
        if (i_bcn == 1 || i_bcn == 3 || i_bcn == 5) {
            f(1, n - 1) = 2.0 * f(3, n - 2);
            f(2, n - 1) = -(f(0, n - 1) - f(0, n - 2)) / f(3, n - 2) + an;
        }
        else if (i_bcn == 2 || i_bcn == 4 || i_bcn == 6) {
            f(1, n - 1) = 2.0 * f(3, n - 2);
            f(2, n - 1) = f(3, n - 2) * bn / 3.0;
            elemnn1 = 0.0;
        }
        else if (i_bcn == 7) {
            f(1, n - 1) = -f(3, n - 2);
            f(2, n - 1) = f(2, n - 2) / (x[n - 1] - x[n - 3]) - f(2, n - 3) / (x[n - 2] - x[n - 4]);
            f(2, n - 1) = -f(2, n - 1) * f(3, n - 2) * f(3, n - 2) / (x[n - 1] - x[n - 4]);
        }
        else if (i_bc1 != -1) {
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
                f(2, i) = (f(2, i) - f(3, i) * f(2, i + 1) - wk[i] * f(2, n - 2)) / f(1, i);
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

                if ((i == imin) && (imin == 1)) {
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
                f(2, n - 1) = f(2, n - 2) + (f(2, n - 2) - f(2, n - 3)) * f(3, n - 2) / f(3, n - 3);
            }
        }

        // Polynomial coefficient computation
        for (int i = 0; i < n - 1; ++i) {
            f(1, i) = (f(0, i + 1) - f(0, i)) / f(3, i) - f(3, i) * (f(2, i + 1) + 2.0 * f(2, i));
            f(3, i) = (f(2, i + 1) - f(2, i)) / f(3, i);
            f(2, i) *= 6.0;
            f(3, i) *= 6.0;
        }

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
void CubicSplineInterpolator<T, MemorySpace>::splinck(Rank1View<const T, MemorySpace>& x, const double& ztol) {
    int inx = static_cast<int>(x.extent(0));

    if (inx <= 1) return;

    double dxavg = (x[inx - 1] - x[0]) / inx;
    double zeps = std::abs(ztol * dxavg);

    for (int ix = 0; ix < inx; ++ix) {
        double zdiffx = x[ix] - x[ix - 1];
        if (zdiffx <= 0.0) {
            throw std::runtime_error("Not strictly ascending");
        }
        double zdiff = zdiffx - dxavg;
        if (std::abs(zdiff) > zeps) {
            throw std::runtime_error("Input is not a uniform grid");
        }
    }
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::spvec(
    const std::vector<int>& selector, // ict[0] = f, ict[1] = df/dx, ict[2] = d²f/dx²
    const int& ivec, // number of x values
    Rank1View<const T, MemorySpace> xvec, // input x vector
    const int& ivd, // output dimension >= ivec
    Rank2View<T, MemorySpace> fval, // output values: fval(ivec, up to 3)
    const int& nx, // number of x grid points
    Rank2View<T, MemorySpace> xpkg, // x package: xpkg(nx, 4)
    Rank2View<T, MemorySpace> fspl, // spline coefficients: fspl(4, nx)
    int& iwarn // warning flag
) {
    // TODO: Allocate Local variables
    std::vector<int> iv(ivec, 0);
    std::vector<double> dxv(ivec, 0.0);
    std::vector<double> dhv(ivec, 0.0);
    std::vector<double> dhiv(ivec, 0.0);

    iwarn = 0;

    if (nx < 2) {
        throw std::runtime_error("nx >= 2 required.");
    }

    if (ivec <= 0) {
        throw std::runtime_error("vector dimension > 0 required.");
    }

    if (ivd < ivec) {
        throw std::runtime_error("output vector dimension less than input vector dimension.");
    }

    // Vectorized lookup
    xlookup(ivec, xvec, nx, xpkg, 1, iv, dxv, dhv, dhiv, iwarn);

    // Vectorized spline evaluation
    cspevfn(selector, ivec, ivd, fval, iv, dxv, fspl, nx);
} // end spvec

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::genxpkg(
    int nx,
    Rank1View<const T, MemorySpace> x,
    Rank2View<T, MemorySpace> xpkg,
    int iper,
    int imsg,
    const int& itol,
    const double& ztol,
    const int& ialg
) {
    if (nx < 2) {
        throw std::runtime_error("nx >= 2 required.");
    }

    int ialgu = ialg;
    if (ialgu == 0) ialgu = 3;
    if (std::abs(ialgu) > 3) ialgu = 3;

    double ztolr = (itol == 1) ? std::abs(ztol) : 5.0e-7;
    double ztola = std::max(std::abs(x[0]), std::abs(x[nx - 1])) * ztolr;

    if (nx >= 3) {
        xpkg(2, 3) = 0.0;
    }

    xpkg(1, 3) = (iper == 1) ? 1.0 : 0.0;
    xpkg(0, 3) = ztola;

    if (imsg != 1) {
        xpkg(0, 3) = -xpkg(0, 3);
    }

    xpkg(nx - 1, 1) = (x[nx - 1] - x[0]) / (nx - 1);  // average spacing

    for (int ix = 0; ix < nx; ++ix) {
        xpkg(ix, 0) = x[ix];
        if (ix < nx - 1) {
            if (x[ix + 1] <= x[ix]) {
                throw std::runtime_error("x axis not strict ascending!");
            } else {
                double zh = x[ix + 1] - x[ix];
                if (nx >= 3 && std::abs(zh - xpkg(nx - 1, 1)) > ztola) {
                    xpkg(2, 3) = 1.0;
                }
                xpkg(ix, 1) = zh;
                xpkg(ix, 2) = 1.0 / zh;
            }
        }
    }


    xpkg(nx - 1, 2) = 1.0 / xpkg(nx - 1, 1);

    if (nx >= 3) {
        if (xpkg(2, 3) == 0.0) {
            for (int ix = 0; ix < nx - 2; ++ix) {
                int ixp = ix + 1;
                xpkg(ixp, 0) = xpkg(0, 0) + ixp * xpkg(nx - 1, 1);
            }

            if (nx > 3) {
                xpkg(3, 3) = (ialgu < 0) ? 1.0 : 0.0;
            }
        }

        if (xpkg(2, 3) != 0.0) {
            xpkg(2, 3) = std::abs(ialgu);
            if (nx > 3) {
                xpkg(3, 3) = (ialgu < 0) ? 1.0 : 0.0;
            }

            if (std::abs(ialgu) == 3) {
                xpkg(0, 1) = 1.0;
                double xtest = xpkg(0, 0);
                int itest = 0;

                for (int i = 1; i < nx - 1; ++i) {
                    xtest += xpkg(nx - 1, 1);
                    while (!(xpkg(itest, 0) <= xtest && xtest <= xpkg(itest + 1, 0))) {
                        ++itest;
                    }
                    // TODO: test if the index is correct
                    xpkg(i, 1) = itest + (xtest - xpkg(itest, 0)) /
                                        (xpkg(itest + 1, 0) - xpkg(itest, 0));
                }
            }
        }
    }
} // end genxpkg

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::cspevfn(
    const std::vector<int>& selector,
    const int& ivec,
    const int& ivd,
    Rank2View<T, MemorySpace> fval,
    std::vector<int>& iv,
    std::vector<double>& dxv,
    Rank2View<T, MemorySpace> f,
    int nx
) {

    int iaval = 0;

    if (selector[0] == 3) {
        // Third derivative only
        iaval++;
        for (int v = 0; v < ivec; ++v) {
            int i = iv[v] - 1;
            fval(v, iaval - 1) = 6.0 * f(3, i);
        }
    } else {
        if (selector[0] > 0) {
            // Evaluate f
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v] - 1;
                double dx = dxv[v];
                fval(v, iaval - 1) = f(0, i) + dx * (f(1, i) + dx * (f(2, i) + dx * f(3, i)));
            }
        }

        if (selector[1] > 0) {
            // Evaluate df/dx
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v] - 1;
                double dx = dxv[v];
                fval(v, iaval - 1) = f(1, i) + dx * (2.0 * f(2, i) + dx * 3.0 * f(3, i));
            }
        }

        if (selector[2] > 0) {
            // Evaluate d2f/dx2
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v] - 1;
                double dx = dxv[v];
                fval(v, iaval - 1) = 2.0 * f(2, i) + dx * 6.0 * f(3, i);
            }
        }
    }
} // end cspevfn

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::spgrid(
    Rank1View<const T, MemorySpace>& x_newgrid,
    const int& nx_new,
    Rank2View<T, MemorySpace>& f_new,
    const int& nx,
    Rank2View<T, MemorySpace>& xpkg,
    Rank2View<T, MemorySpace>& fspl,
    int& iwarn
){
    std::vector<int> ict = {1, 0, 0};
    spvec(ict, nx_new, x_newgrid, nx_new, f_new, nx, xpkg, fspl, iwarn);
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::xlookup(
    const int& ivec,
    Rank1View<const T, MemorySpace> xvec,
    const int& nx,
    Rank2View<T, MemorySpace> xpkg,
    const int& imode,
    std::vector<int>& iv,
    std::vector<double>& dxn,
    std::vector<double>& hv,
    std::vector<double>& hiv,
    int& iwarn
) {
    iwarn = 0;
    // Check if grid size is valid
    if (nx < 2) {
        iwarn = 1;
        std::cerr << "xlookup: nx < 2, nx = " << nx << std::endl;
        return;
    }

    int ilin, ialg, iper, imsg, init_guess, iprev;
    double ztola;
    // TODO: Allocate xuse, probably initialize a View
    std::vector<T> xuse(ivec);

    // Determine if grid is linear or irregular
    if (nx == 2) {
        ilin = 1;
    }

    if (nx > 2) {
        if (xpkg(2, 3) == 0.0) {
            ilin = 1;  // evenly spaced grid
        } else {
            ilin = 0;
            if (xpkg(2, 3) > 2.5) {
                ialg = 3;
            } else if (xpkg(2, 3) > 1.5) {
                ialg = 2;
            } else {
                ialg = 1;
            }
        }
    }

    // Check for periodic grid
    iper = (xpkg(1, 3) != 0.0) ? 1 : 0;

    // Range checking tolerance
    ztola = std::abs(xpkg(0, 3));

    // Message flag on range error
    imsg = (xpkg(0, 3) >= 0.0) ? 1 : 0;

    // Initial guess optimization
    init_guess = 0;
    if (nx > 3) {
        if (xpkg(3, 3) > 0.0) {
            init_guess = 1;
            iprev = std::min(nx - 1, std::max(1, iv[ivec - 1]));
        }
    }

    // Final status
    iwarn = 0;

    if (iper == 0) {
        // Non-periodic: clamp with tolerance
        for (int i = 0; i < ivec; ++i) {
            if (xvec[i] < xpkg(0, 0)) {
                xuse[i] = xpkg(0, 0);
                if ((xpkg(0, 0) - xvec[i]) > ztola) {
                    ++iwarn;
                }
            } else if (xvec[i] > xpkg(nx - 1, 0)) {
                xuse[i] = xpkg(nx - 1, 0);
                if ((xvec[i] - xpkg(nx - 1, 0)) > ztola) {
                    ++iwarn;
                }
            } else {
                xuse[i] = xvec[i];
            }
        }
    } else {
        // Periodic: normalize to interval
        double period = xpkg(nx - 1, 0) - xpkg(0, 0);
        for (int i = 0; i < ivec; ++i) {
            if (xvec[i] < xpkg(0, 0) || xvec[i] > xpkg(nx - 1, 0)) {
                double shifted = std::fmod(xvec[i] - xpkg(0, 0), period);
                if (shifted < 0.0) shifted += period;
                xuse[i] = std::clamp(shifted + xpkg(0, 0), xpkg(0, 0), xpkg(nx - 1, 0));
            } else {
                xuse[i] = xvec[i];
            }
        }
    }

    if (imsg == 1 && iwarn > 0) {
        std::cerr << "lookup: points not in range: "
                << xpkg(0, 0) << " to " << xpkg(nx - 1, 0) << std::endl;
    }

    // Grid spacing and its inverse for linear assumption (used later)
    double hav = xpkg(nx - 1, 1);
    double havi = xpkg(nx - 1, 2);

    // (you can return hav, havi, or continue processing here)
    if (ilin == 1) {
        if (init_guess == 0) {
            // Even spacing without initial guess
            for (int i = 0; i < ivec; ++i) {
                iv[i] = 1 + static_cast<int>(havi * (xuse[i] - xpkg(0, 0)));
                iv[i] = std::max(1, std::min(nx - 1, iv[i]));
                if (imode == 1) {
                    dxn[i] = xuse[i] - xpkg(iv[i] - 1, 0);
                } else {
                    dxn[i] = (xuse[i] - xpkg(iv[i] - 1, 0)) * havi;
                    hiv[i] = havi;
                    hv[i] = hav;
                }
            }
        } else {
            // Even spacing with initial guess
            // TODO: make sure this part is correct
            for (int i = 0; i < ivec; ++i) {
                if (xpkg(iprev - 1, 0) <= xuse[i] && xuse[i] <= xpkg(iprev, 0)) {
                    iv[i] = iprev;
                } else {
                    iv[i] = 1 + static_cast<int>(havi * (xuse[i] - xpkg(0, 0)));
                    iv[i] = std::max(1, std::min(nx - 1, iv[i]));
                    iprev = iv[i];
                }

                if (imode == 1) {
                    dxn[i] = xuse[i] - xpkg(iv[i] - 1, 0);
                } else {
                    dxn[i] = (xuse[i] - xpkg(iv[i] - 1, 0)) * havi;
                    hiv[i] = havi;
                    hv[i] = hav;
                }
            }
        }
    }
    // TODO: deallocate xuse
} // end xlookup








} //namespace pcms
#endif