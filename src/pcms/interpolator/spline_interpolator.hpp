
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

template <typename ElementType, typename MemorySpace>
using Rank3View = View<3, ElementType, MemorySpace>;

template <typename ElementType, typename MemorySpace>
using Rank4View = View<4, ElementType, MemorySpace>;

template <typename T, typename MemorySpace>
class SplineInterpolator {
public:
    SplineInterpolator() = default;
    void splinck(Rank1View<const T, MemorySpace> x, const double& ztol);
};

template <typename T, typename MemorySpace>
void SplineInterpolator<T, MemorySpace>::splinck(Rank1View<const T, MemorySpace> x, const double& ztol) {
    int inx = static_cast<int>(x.extent(0));

    if (inx <= 1) return;

    double dxavg = (x[inx - 1] - x[0]) / (inx - 1);
    double zeps = std::abs(ztol * dxavg);

    for (int ix = 1; ix < inx; ++ix) {
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



// TODO: better error handling, use pcms existing error handling
// TODO: better documentation
// TODO: add more checks/assetions for input parameters
// TODO: remove the usage of size variable if it is better to use extent(0) or extent(1) in mdspan
template <typename T, typename MemorySpace>
class CubicSplineInterpolator: public SplineInterpolator<T, MemorySpace> {
public:

    CubicSplineInterpolator() = default;

    void cspline(Rank1View<const T, MemorySpace> x, int& nx, Rank2View<T, MemorySpace> fspl, const int& ibcxmin,
        const double& bcxmin, const int& ibcxmax, const double& bcxmax,
        Rank1View<T, MemorySpace> wk, const int& iwk);

    void v_spline(const int& k_bc1, const int& k_bcn, const int& n, Rank1View<const T, MemorySpace> x,
        Rank2View<T, MemorySpace> f, Rank1View<T, MemorySpace> wk);

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

    void mkspline(
        Rank1View<const T, MemorySpace> x,
        int& nx,
        Rank2View<T, MemorySpace> fspl,
        Rank2View<T, MemorySpace> fspl4,
        const int& ibcxmin,
        const double& bcxmin,
        const int& ibcxmax,
        const double& bcxmax,
        Rank1View<T, MemorySpace> wk
    );

    void vecspline(const std::vector<int>& selector, int ivec,
        Rank1View<const T, MemorySpace>, int ivd,
        Rank2View<T, MemorySpace> fval,
        int nx, Rank2View<T, MemorySpace> xpkg,
        Rank2View<T, MemorySpace> fspl,
        int& iwarn);

    void fvspline(const std::vector<int>& selector, int ivec, int ivecd,
        Rank2View<T, MemorySpace> fval,
        const std::vector<int>& ii,
        const std::vector<double>& xparam,
        const std::vector<double>& hx,
        const std::vector<double>& hxi,
        Rank2View<T, MemorySpace> fin,
        int nx);

    void gridspline(Rank1View<const T, MemorySpace> x_newgrid, int nx_new,
        Rank2View<T, MemorySpace> f_new,
        int nx,
        Rank2View<T, MemorySpace> xpkg,
        Rank2View<T, MemorySpace> fspl,
        int& iwarn);
    
    void cspeval(double xget, const std::vector<int>& iselect,
        Rank2View<T, MemorySpace> fval,
        Rank1View<const T, MemorySpace> x, int nx,
        Rank2View<T, MemorySpace> f, int& ier);
    
    void cspevx(double xget, Rank1View<const T, MemorySpace> x, const int& nx,
        int& i, double& dx, int& ier);
    
    void evspline(double xget,
        const std::vector<int>& ict,
        Rank2View<T, MemorySpace> fval,
        Rank1View<const T, MemorySpace> x,
        int& nx,
        Rank2View<T, MemorySpace> f,
        int& ier
    );
    
    void herm1x(double xget,
        Rank1View<const T, MemorySpace> x,
        const int & nx,
        int& i,
        double& xparam,
        double& hx,
        double& hxi,
        int& ier);

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
            int i = iv[v];
            fval(v, iaval - 1) = 6.0 * f(3, i);
        }
    } else {
        if (selector[0] > 0) {
            // Evaluate f
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                double dx = dxv[v];
                fval(v, iaval - 1) = f(0, i) + dx * (f(1, i) + dx * (f(2, i) + dx * f(3, i)));
            }
        }

        if (selector[1] > 0) {
            // Evaluate df/dx
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                double dx = dxv[v];
                fval(v, iaval - 1) = f(1, i) + dx * (2.0 * f(2, i) + dx * 3.0 * f(3, i));
            }
        }

        if (selector[2] > 0) {
            // Evaluate d2f/dx2
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
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
            iprev = std::min(nx - 2, std::max(0, iv[ivec - 1]));
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
            // We can optimize this case by correcting indexes directly
            for (int i = 0; i < ivec; ++i) {
                iv[i] = static_cast<int>(havi * (xuse[i] - xpkg(0, 0)));
                iv[i] = std::max(0, std::min(nx - 2, iv[i]));
                if (imode == 1) {
                    dxn[i] = xuse[i] - xpkg(iv[i], 0);
                } else {
                    dxn[i] = (xuse[i] - xpkg(iv[i], 0)) * havi;
                    hiv[i] = havi;
                    hv[i] = hav;
                }
            }
        } else {
            // Even spacing with initial guess
            // TODO: make sure this part is correct
            for (int i = 0; i < ivec; ++i) {
                if (xpkg(iprev, 0) <= xuse[i] && xuse[i] <= xpkg(iprev + 1, 0)) {
                    iv[i] = iprev;
                } else {
                    iv[i] = static_cast<int>(havi * (xuse[i] - xpkg(0, 0)));
                    iv[i] = std::max(0, std::min(nx - 2, iv[i]));
                    iprev = iv[i];
                }

                if (imode == 1) {
                    dxn[i] = xuse[i] - xpkg(iv[i], 0);
                } else {
                    dxn[i] = (xuse[i] - xpkg(iv[i], 0)) * havi;
                    hiv[i] = havi;
                    hv[i] = hav;
                }
            }
        }
    }
    // TODO: deallocate xuse
} // end xlookup

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::cspeval(double xget, const std::vector<int>& iselect,
             Rank2View<T, MemorySpace> fval,
             Rank1View<const T, MemorySpace> x, int nx,
             Rank2View<T, MemorySpace> f,
             int& ier) {
    std::vector<int> ia(1, 0);
    std::vector<double> dxa(1, 0);
    cspevx(xget, x, nx, ia[0], dxa[0], ier);
    if (ier != 0) {
        std::cerr << "cspeval: error in cspevx, ier = " << ier << std::endl;
        return;
    }

    cspevfn(iselect, 1, 1, fval, ia, dxa, f, nx);
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::cspevx(double xget, Rank1View<const T, MemorySpace> x, const int& nx,
            int& i, double& dx, int& ier) {
    int nxm = nx - 1;
    double zxget = xget;

    // Range check
    if (xget < x[0] || xget > x[nx - 1]) {
        double zxtol = 4.0e-7 * std::max(std::abs(x[0]), std::abs(x[nx - 1]));

        if (xget < x[0] - zxtol || xget > x[nx - 1] + zxtol) {
            ier = 1; // Error code for out of range
            std::cerr << "cspeval: xget = " << xget
                      << " out of range " << x[0] << " to " << x[nx - 1] << "\n";
            return;
        } else {
            std::cerr << "cspeval: xget = " << xget
                      << " beyond range " << x[0] << " to " << x[nx - 1]
                      << " (fixup applied)\n";
            zxget = (xget < x[0]) ? x[0] : x[nx - 1];
        }
    }


    int ii = static_cast<int>(nxm * (zxget - x[0]) / (x[nx - 1] - x[0]));
    i = std::min(nxm - 1, ii);  // Fortran is 1-based, C++ is 0-based

    if (zxget < x[i]) {
        i = std::max(0, i - 1);
    } else if (zxget > x[i + 1]) {
        i = std::min(nxm - 1, i + 1);
    }

    dx = zxget - x[i];
}



void ibc_ck(int ibc, const std::string& slbl, const std::string& xlbl,
            int imin, int imax, int& ier) {
    if (ibc < imin || ibc > imax) {
        ier = 1;
        std::cerr << " ? " << slbl << " -- ibc" << xlbl
                  << " = " << ibc << " out of range "
                  << imin << " to " << imax << std::endl;
    }
}


template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::mkspline(
    Rank1View<const T, MemorySpace> x,
    int& nx,
    Rank2View<T, MemorySpace> fspl,
    Rank2View<T, MemorySpace> fspl4,
    const int& ibcxmin,
    const double& bcxmin,
    const int& ibcxmax,
    const double& bcxmax,
    Rank1View<T, MemorySpace> wk
){
    // TODO: input check, size check and ascending

    // Copy f data to fspl4 and zero out second derivative output
    for (int i = 0; i < nx; ++i) {
        fspl4(0, i) = fspl(0, i); // f(1,*) → fspl4(1,*)
        fspl(1, i) = 0.0;         // f(2,*) = 0.0 initially
    }

    int inwk = nx;

    // Call traditional spline generator
    cspline(x, nx, fspl4, ibcxmin, bcxmin, ibcxmax, bcxmax, wk, inwk);
    
    for (int i = 0; i < nx - 1; ++i) {
        fspl(1, i) = 2.0 * fspl4(2, i); // fspl(2,*) = 2 * fspl4(3,*)
    }
    fspl(1, nx - 1) = 2.0 * fspl4(2, nx - 2) +
                        (x[nx - 1] - x[nx - 2]) * 6.0 * fspl4(3, nx - 2);
    
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::vecspline(const std::vector<int>& selector, int ivec,
               Rank1View<const T, MemorySpace> xvec, int ivd,
               Rank2View<T, MemorySpace> fval,
               int nx, Rank2View<T, MemorySpace> xpkg,
               Rank2View<T, MemorySpace> fspl,
               int& iwarn)
{
    std::vector<int> iv(ivec, 0);
    std::vector<double> dxn(ivec, 0.0);
    std::vector<double> h(ivec, 0.0);
    std::vector<double> hi(ivec, 0.0);

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

    // Call vectorized lookup
    xlookup(ivec, xvec, nx, xpkg, 2, iv, dxn, h, hi, iwarn);

    // Call vectorized spline evaluation
    fvspline(selector, ivec, ivd, fval, iv, dxn, h, hi, fspl, nx);
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::fvspline(const std::vector<int>& selector, int ivec, int ivecd,
              Rank2View<T, MemorySpace> fval,
              const std::vector<int>& ii,
              const std::vector<double>& xparam,
              const std::vector<double>& hx,
              const std::vector<double>& hxi,
              Rank2View<T, MemorySpace> fin,
              int nx)
{
    const double sixth = 1.0 / 6.0;
    int iadr = 0;

    if (selector[0] <= 2) {
        if (selector[0] == 1) {
            // Function value f(x)
            ++iadr;
            for (int v = 0; v < ivec; ++v) {
                //TODO: too much temp varibale created?
                int i = ii[v];
                double xp = xparam[v];
                double xpi = 1.0 - xp;
                double xp2 = xp * xp;
                double xpi2 = xpi * xpi;
                double cx = xp * (xp2 - 1.0);
                double cxi = xpi * (xpi2 - 1.0);
                double hx2 = hx[v] * hx[v];

                double sum = xpi * fin(0, i) + xp * fin(0, i + 1);
                sum += sixth * hx2 * (cxi * fin(1, i) + cx * fin(1, i + 1));

                fval(v, iadr - 1) = sum; // zero-based indexing
            }
        }

        if (selector[1] == 1) {
            // First derivative df/dx
            ++iadr;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v];
                double xp = xparam[v];
                double xpi = 1.0 - xp;
                double xp2 = xp * xp;
                double xpi2 = xpi * xpi;

                double cxd = 3.0 * xp2 - 1.0;
                double cxdi = -3.0 * xpi2 + 1.0;

                double sum = hxi[v] * (fin(0, i + 1) - fin(0, i));
                sum += sixth * hx[v] * (cxdi * fin(1, i) + cxd * fin(1, i + 1));

                fval(v, iadr - 1) = sum;
            }
        }

        if (selector[2] == 1) {
            // Second derivative d2f/dx2
            ++iadr;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v];
                double xp = xparam[v];
                double xpi = 1.0 - xp;

                double sum = xpi * fin(1, i) + xp * fin(1, i + 1);
                fval(v, iadr - 1) = sum;
            }
        }

    } else {
        // Third derivative d3f/dx3
        iadr = 1;
        for (int v = 0; v < ivec; ++v) {
            int i = ii[v];
            fval(v, iadr - 1) = hxi[v] * (fin(1, i + 1) - fin(1, i));
        }
    }
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::gridspline(Rank1View<const T, MemorySpace> x_newgrid, int nx_new,
                Rank2View<T, MemorySpace> f_new,
                int nx,
                Rank2View<T, MemorySpace> xpkg,
                Rank2View<T, MemorySpace> fspl,
                int& iwarn)
{
    // Request only function value (no derivatives)
    std::vector<int> ict = {1, 0, 0};
    // Call vecspline
    vecspline(ict, nx_new, x_newgrid, nx_new, f_new, nx, xpkg, fspl, iwarn);
}

template <typename T, typename MemorySpace>
void CubicSplineInterpolator<T, MemorySpace>::herm1x(double xget,
            Rank1View<const T, MemorySpace> x,
            const int & nx,
            int& i,
            double& xparam,
            double& hx,
            double& hxi,
            int& ier) {
    ier = 0;

    double zxget = xget;

    if ((xget < x[0]) || (xget > x[nx - 1])) {
        double zxtol = 4.0e-7 * std::max(std::abs(x[0]), std::abs(x[nx - 1]));
        if ((xget < x[0] - zxtol) || (xget > x[nx - 1] + zxtol)) {
            ier = 1;
            std::cerr << "herm1ev:  xget=" << xget
                      << " out of range " << x[0] << " to " << x[nx - 1] << "\n";
            return;
        } else {
            if ((xget < x[0] - 0.5 * zxtol) || (xget > x[nx - 1] + 0.5 * zxtol)) {
                std::cerr << "herm1ev:  xget=" << xget
                          << " beyond range " << x[0] << " to " << x[nx - 1]
                          << " (fixup applied)\n";
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
void CubicSplineInterpolator<T, MemorySpace>::evspline(double xget,
        const std::vector<int>& ict,
        Rank2View<T, MemorySpace> fval,
        Rank1View<const T, MemorySpace> x,
        int& nx,
        Rank2View<T, MemorySpace> f, // shape: [2, nx]
        int& ier
    ) {

    // Initialize output zone info
    std::vector<int> i(1, 0);
    std::vector<double> xparam(1);
    std::vector<double> hx(1);
    std::vector<double> hxi(1);

    // Find the interval containing xget
    herm1x(xget, x, nx, i[0], xparam[0], hx[0], hxi[0], ier);
    if (ier != 0) return;

    // Evaluate spline at the point
    fvspline(ict, 1, 1, fval, i, xparam, hx, hxi, f, nx);
}


template <typename T, typename MemorySpace>
class BiCubicSplineInterpolator: public CubicSplineInterpolator<T, MemorySpace> {
public:

    BiCubicSplineInterpolator() = default;
    void bcspline(
        Rank1View<const T, MemorySpace>x,             // size: inx
        int inx,
        Rank1View<const T, MemorySpace> th,            // size: inth
        int inth,
        Rank4View<T, MemorySpace> fspl, // [4, 4, inf3, inth]
        int inf3,
        int ibcxmin,
        Rank1View<T, MemorySpace> bcxmin,        // size: inth (used if ibcxmin = 1 or 2)
        int ibcxmax,
        Rank1View<T, MemorySpace> bcxmax,        // size: inth (used if ibcxmax = 1 or 2)
        int ibcthmin,
        Rank1View<T, MemorySpace> bcthmin,       // size: inx (used if ibcthmin = 1 or 2)
        int ibcthmax,
        Rank1View<T, MemorySpace> bcthmax,       // size: inx (used if ibcthmax = 1 or 2)
        Rank1View<T, MemorySpace> wk,                  // size: nwk
        int nwk,
        int& ilinx,                           // output: 1 if x is evenly spaced
        int& ilinth,                          // output: 1 if th is evenly spaced
        int& ier                              // output: error code
    );

    static void bcspeval(double xget, double yget,
              const std::vector<int>& iselect,
              Rank2View<double, HostMemorySpace> fval,
              Rank1View<const double, HostMemorySpace> x, int nx,
              Rank1View<const double, HostMemorySpace> y, int ny,
              int ilinx, int iliny,
              Rank4View<double, HostMemorySpace> f,
              int inf3, int& ier);

    static void bcspevxy(
        double xget, double yget,
        Rank1View<const double, HostMemorySpace> x, int nx,
        Rank1View<const double, HostMemorySpace> y, int ny,
        int ilinx, int iliny,
        int& i, int& j,
        double& dx, double& dy,
        int& ier
    );

    static void bcspevfn(
        const std::vector<int>& ict,             // Selector array for which derivatives to compute
        int ivec,                     // Number of vector points
        int ivd,                      // First dimension of fval (≥ ivec)
        Rank2View<T, MemorySpace> fval,                 // Output array: size [ivd, *] (flattened)
        std::vector<int>& iv,                // Grid cell indices in x direction
        std::vector<int>& jv,                // Grid cell indices in y direction
        std::vector<double>& dxv,            // x displacements within cells
        std::vector<double>& dyv,            // y displacements within cells
        Rank4View<T, MemorySpace> f,              // Bicubic spline coefficients [4, 4, inf3, ny] (flattened)
        int inf3,                     // 3rd dimension of f
        int ny                        // Number of y points (4th dim of f)
    );

    void mkbicub(
        Rank1View<const T, MemorySpace> x, int nx,
        Rank1View<const T, MemorySpace> y, int ny,
        Rank3View<T, MemorySpace> f, int nf2,
        int ibcxmin, Rank1View<T, MemorySpace> bcxmin,
        int ibcxmax, Rank1View<T, MemorySpace> bcxmax,
        int ibcymin, Rank1View<T, MemorySpace> bcymin,
        int ibcymax, Rank1View<T, MemorySpace> bcymax,
        int& ilinx, int& iliny, int& ier
    );

    static void herm2xy(double xget, double yget,
                    Rank1View<const T, MemorySpace> x,
                    int& nx,
                    Rank1View<const T, MemorySpace> y,
                    int& ny,
                    bool ilinx, bool iliny,
                    int& i, int& j,
                    double& xparam, double& yparam,
                    double& hx, double& hxi,
                    double& hy, double& hyi,
                    int& ier);

    static void fvbicub(const std::vector<int>& ict, int ivec, int ivecd,
                    Rank2View<T, MemorySpace> fval,
                    const std::vector<int>& ii,
                    const std::vector<int>& jj,
                    const std::vector<double>& xparam,
                    const std::vector<double>& yparam,
                    const std::vector<double>& hx,
                    const std::vector<double>& hxi,
                    const std::vector<double>& hy,
                    const std::vector<double>& hyi,
                    Rank3View<T, MemorySpace> fin);
    
    static void evbicub(double xget, double yget,
        Rank1View<const T, MemorySpace> x, int nx,
        Rank1View<const T, MemorySpace> y, int ny,
        int ilinx, int iliny,
        Rank3View<T, MemorySpace> f, // f[4][inf2][ny]
        int inf2,
        const std::vector<int>& ict,
        Rank2View<T, MemorySpace> fval, // output (size depends on ict)
        int& ier);
    
};

template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::bcspline(
    Rank1View<const T, MemorySpace> x,             // size: inx
    int inx,
    Rank1View<const T, MemorySpace> th,            // size: inth
    int inth,
    Rank4View<T, MemorySpace> fspl, // [4, 4, inf3, inth]
    int inf3,
    int ibcxmin,
    Rank1View<T, MemorySpace> bcxmin,        // size: inth (used if ibcxmin = 1 or 2)
    int ibcxmax,
    Rank1View<T, MemorySpace> bcxmax,        // size: inth (used if ibcxmax = 1 or 2)
    int ibcthmin,
    Rank1View<T, MemorySpace> bcthmin,       // size: inx (used if ibcthmin = 1 or 2)
    int ibcthmax,
    Rank1View<T, MemorySpace> bcthmax,       // size: inx (used if ibcthmax = 1 or 2)
    Rank1View<T, MemorySpace> wk,                  // size: nwk
    int nwk,
    int& ilinx,                           // output: 1 if x is evenly spaced
    int& ilinth,                          // output: 1 if th is evenly spaced
    int& ier                              // output: error code
) {
    int iflg2 = 0;
    std::vector<int> iselect1(10, 0); // Size 10, all initialized to 0
    std::vector<int> iselect2(10, 0); // Size 10, all initialized to 0
    std::vector<double> zcur_vec(3, 0.0); // Size 1, initialized to 0.0
    auto fspl_x = Rank2View<T, MemorySpace>(wk.data_handle() + 4 * inx * inth, 4, inx);
    auto fspl_th = Rank2View<T, MemorySpace>(wk.data_handle() + 4 * inx * inth, 4, inth);
    auto fspl_l_th = Rank2View<T, MemorySpace>(wk.data_handle(), 4 * inx, inth);
    auto wk_x = Rank1View<T, MemorySpace>(wk.data_handle() + 4 * inx * inth + 4 * inx, inx);
    auto wk_th = Rank1View<T, MemorySpace>(wk.data_handle() + 4 * inx * inth + 4 * inth, inth);

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

    ier = 0;
    int itest = 5 * std::max(inx, inth);
    if (iflg2 == 1) {
        itest += 4 * inx * inth;
    }

    if (nwk < itest) {
        std::cerr << " bcspline:  workspace too small:\n"
                << "  user supplied:  nwk=" << nwk
                << "; need at least:  " << itest << "\n"
                << "  nwk=4*nx*ny +5*max(nx,ny) will work for any user\n"
                << "  choice of bdy conditions.\n";
        return;
    }

    if (inx < 2) {
        std::cerr << " bcspline:  at least 2 x points required.\n";
        return;
    }

    if (inth < 2) {
        std::cerr << " bcspline:  need at least 2 theta points.\n";
        return;
    }

    // Check boundary condition values
    ibc_ck(ibcxmin, "bcspline", "xmin", -1, 7, ier);
    if (ibcxmin >= 0) ibc_ck(ibcxmax, "bcspline", "xmax", 0, 7, ier);
    ibc_ck(ibcthmin, "bcspline", "thmin", -1, 7, ier);
    if (ibcthmin >= 0) ibc_ck(ibcthmax, "bcspline", "thmax", 0, 7, ier);

    // Check x vector spacing
    int ierx = 0;
    CubicSplineInterpolator<T, MemorySpace>::splinck(x, 1.0e-3);
    if (ierx != 0) ier = 2;
    if (ier == 2) {
        std::cerr << " bcspline:  x axis not strict ascending\n";
        return;
    }

    // Check th vector spacing
    int ierth = 0;
    CubicSplineInterpolator<T, MemorySpace>::splinck(th, 1.0e-3);
    if (ierth != 0) ier = 3;
    if (ier == 3) {
        std::cerr << " bcspline:  th axis not strict ascending\n";
        return;
    }

    if (ier != 0) return;

    double xo2 = 0.5;
    double xo6 = 1.0 / 6.0;

    // Spline in x-direction
    int inxo = 4 * (inx - 1);

    for (int ith = 0; ith < inth; ++ith) {
        // Copy function into workspace
        for (int ix = 0; ix < inx; ++ix) {
            fspl_x(0, ix) = fspl(0, 0, ix, ith); // fspl(1,1,ix,ith)
        }

        // Boundary condition at xmin
        if (ibcxmin == 1) {
            fspl_x(1, 0) = bcxmin[ith];
        } else if (ibcxmin == 2) {
            fspl_x(2, 0) = bcxmin[ith];
        }

        // Boundary condition at xmax
        if (ibcxmax == 1) {
            fspl_x(1, inx - 1) = bcxmax[ith];
        } else if (ibcxmax == 2) {
            fspl_x(2, inx - 1) = bcxmax[ith];
        }

        // Call v_spline
        BiCubicSplineInterpolator<T, MemorySpace>::v_spline(ibcxmin, ibcxmax, inx, x, fspl_x, wk_x);

        // Copy coefficients out
        for (int ix = 0; ix < inx; ++ix) {
            fspl(1, 0, ix, ith) = fspl_x(1, ix);               // fspl(2,1,...)
            fspl(2, 0, ix, ith) = fspl_x(2, ix) * xo2;          // fspl(3,1,...)
            fspl(3, 0, ix, ith) = fspl_x(3, ix) * xo6;          // fspl(4,1,...)
        }
    }

    // Spline in theta direction
    int intho = 4 * (inth - 1);

    for (int ix = 0; ix < inx; ++ix) {
        // Spline each x coefficient
        for (int ic = 0; ic < 4; ++ic) {
            // Copy ordinates into workspace
            for (int ith = 0; ith < inth; ++ith) {
                fspl_th(0, ith) = fspl(ic, 0, ix, ith); // fspl(ic,1,ix,ith)
            }

            // Set linear BCs initially
            fspl_th(1, 0) = 0.0;
            fspl_th(2, 0) = 0.0;
            fspl_th(1, inth - 1) = 0.0;
            fspl_th(2, inth - 1) = 0.0;

            // Adjust BC flags if needed
            int ibcthmina = ibcthmin;
            int ibcthmaxa = ibcthmax;
            if (iflg2 == 1) {
                if (ibcthmin == 1 || ibcthmin == 2) ibcthmina = 0;
                if (ibcthmax == 1 || ibcthmax == 2) ibcthmaxa = 0;
            }

            // Call v_spline
            BiCubicSplineInterpolator<T, MemorySpace>::v_spline(ibcthmina, ibcthmaxa, inth, th, fspl_th, wk_th);

            // Copy coefficients out
            for (int ith = 0; ith < inth; ++ith) {
                fspl(ic, 1, ix, ith) = fspl_th(1, ith);              // fspl(ic,2,...)
                fspl(ic, 2, ix, ith) = fspl_th(2, ith) * xo2;         // fspl(ic,3,...)
                fspl(ic, 3, ix, ith) = fspl_th(3, ith) * xo6;         // fspl(ic,4,...)
            }
        }
    }

    if (iflg2 == 1) {
        int iasc = 0;                      // Workspace base for correction splines
        int iinc = 4 * inth;             // Spacing between correction splines
        int iawk = iasc + 4 * inth * inx;

        double zhxn = x[inx - 1] - x[inx - 2];
        int jx = inx - 2;
        double zhth = th[inth - 1] - th[inth - 2];
        int jth = inth - 2;

        for (int ii=0; ii < 10; ++ii) {
            iselect1[ii] = 0;
            iselect2[ii] = 0;
        }

        if (ibcthmin == 1) iselect1[2] = 1;
        if (ibcthmin == 2) iselect1[4] = 1;
        if (ibcthmax == 1) iselect2[2] = 1;
        if (ibcthmax == 2) iselect2[4] = 1;

        for (int ix = 0; ix < inx; ++ix) {
            double zdiff1 = 0.0, zdiff2 = 0.0;

            if (ibcthmin == 1) {
                zcur_vec[0] = (ix < inx - 1) ? fspl(0, 1, ix, 0)
                                        : fspl(0, 1, jx, 0) + zhxn * (fspl(1, 1, jx, 0) + zhxn * (fspl(2, 1, jx, 0) + zhxn * fspl(3, 1, jx, 0)));
                zdiff1 = bcthmin[ix] - zcur_vec[0];
            } else if (ibcthmin == 2) {
                zcur_vec[0] = (ix < inx - 1) ? 2.0 * fspl(0, 2, ix, 0)
                                        : 2.0 * (fspl(0, 2, jx, 0) + zhxn * (fspl(1, 2, jx, 0) + zhxn * (fspl(2, 2, jx, 0) + zhxn * fspl(3, 2, jx, 0))));
                zdiff1 = bcthmin[ix] - zcur_vec[0];
            }

            auto zcur = Rank2View<double, HostMemorySpace>(zcur_vec.data(), 1, 3);
            if (ibcthmax == 1) {
                if (ix < inx - 1) {
                    zcur(0, 0) = fspl(0, 1, ix, jth) + zhth * (2.0 * fspl(0, 2, ix, jth) + zhth * 3.0 * fspl(0, 3, ix, jth));
                } else {
                    // TODO: check if this is correct
                    BiCubicSplineInterpolator<T, MemorySpace>::bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, x, inx, th, inth, ilinx, ilinth, fspl, inf3, ier);
                    if (ier != 0) return;
                }
                zdiff2 = bcthmax[ix] - zcur(0, 0);
            } else if (ibcthmax == 2) {
                if (ix < inx - 1) {
                    zcur(0, 0) = 2.0 * fspl(0, 2, ix, jth) + 6.0 * zhth * fspl(0, 3, ix, jth);
                } else {
                    // TODO: check if this is correct
                    BiCubicSplineInterpolator<T, MemorySpace>::bcspeval(x[inx - 1], th[inth - 1], iselect2, zcur, x, inx, th, inth, ilinx, ilinth, fspl, inf3, ier);
                    if (ier != 0) return;
                }
                zdiff2 = bcthmax[ix] - zcur(0, 0);
            }

            int iadr = iasc + ix * iinc;
            for (int ith = 0; ith < inth; ++ith)
                fspl_l_th(ix * 4, ith) = 0.0;

            fspl_l_th(1 + ix * 4, 0) = 0.0;
            fspl_l_th(2 + ix * 4, 0) = 0.0;
            fspl_l_th(1 + ix * 4, inth - 1) = 0.0;
            fspl_l_th(2 + ix * 4, inth - 1) = 0.0;

            if (ibcthmin == 1) fspl_l_th(1 + ix * 4, 0) = zdiff1;
            else if (ibcthmin == 2) fspl_l_th(2 + ix * 4, 0) = zdiff1;

            if (ibcthmax == 1) fspl_l_th(1 + ix * 4, inth - 1) = zdiff2;
            else if (ibcthmax == 2) fspl_l_th(2 + ix * 4, inth - 1) = zdiff2;

            auto fspl_s_th = Rank2View<double, HostMemorySpace>(fspl_l_th.data_handle() + iadr, 4, inth);
            BiCubicSplineInterpolator<T, MemorySpace>::v_spline(ibcthmin, ibcthmax, inth, th, fspl_s_th, wk_th);

        }

        for (int ix = 0; ix < inx; ++ix) {
            int iadr = iasc + ix * iinc;
            for (int ith = 0; ith < inth - 1; ++ith) {
                fspl_l_th(2 + ix * 4, ith) *= xo2;
                fspl_l_th(3 + ix * 4, ith) *= xo6;
                if (ix < inx - 1) {
                    fspl(0, 1, ix, ith) += fspl_l_th(1 + ix * 4, ith);
                    fspl(0, 2, ix, ith) += fspl_l_th(2 + ix * 4, ith);
                    fspl(0, 3, ix, ith) += fspl_l_th(3 + ix * 4, ith);
                }
            }
        }

        int ia5w = iawk + 4 * inx;

        for (int ith = 0; ith < inth - 1; ++ith) {
            for (int ic = 1; ic < 4; ++ic) {
                for (int ix = 0; ix < inx; ++ix) {
                    int iaspl = iasc + iinc * ix;
                    fspl_x(0, ix) = fspl_l_th(ic + ix * 4, ith);
                }
                fspl_x(1, 0) = 0.0;
                fspl_x(2, 0) = 0.0;
                fspl_x(1, inx - 1) = 0.0;
                fspl_x(2, inx - 1) = 0.0;
                
                BiCubicSplineInterpolator<T, MemorySpace>::v_spline(ibcxmin, ibcxmax, inx, x, fspl_x, wk_x);

                for (int ix = 0; ix < inx - 1; ++ix) {
                    fspl(1, ic, ix, ith) += fspl_x(1, ix);
                    fspl(2, ic, ix, ith) += fspl_x(2, ix) * xo2;
                    fspl(3, ic, ix, ith) += fspl_x(3, ix) * xo6;
                }
            }
        }
    }
}

template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::bcspeval(double xget, double yget,
              const std::vector<int>& iselect,
              Rank2View<double, HostMemorySpace> fval,
              Rank1View<const double, HostMemorySpace> x, int nx,
              Rank1View<const double, HostMemorySpace> y, int ny,
              int ilinx, int iliny,
              Rank4View<double, HostMemorySpace> f,
              int inf3, int& ier) {
    int i = 0;
    int j = 0;
    double dx = 0.0;
    double dy = 0.0;

    // Range finding
    bcspevxy(xget, yget, x, nx, y, ny, ilinx, iliny, i, j, dx, dy, ier);
    if (ier != 0) return;
    std::vector<int> i_vec(1, i); 
    std::vector<int> j_vec(1, j);
    std::vector<double> dx_vec(1, dx);
    std::vector<double> dy_vec(1, dy);

    // Evaluate spline function
    bcspevfn(iselect, 1, 1, fval, i_vec, j_vec, dx_vec, dy_vec, f, inf3, ny);
}

template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::bcspevxy(
    double xget, double yget,
    Rank1View<const double, HostMemorySpace> x, int nx,
    Rank1View<const double, HostMemorySpace> y, int ny,
    int ilinx, int iliny,
    int& i, int& j,
    double& dx, double& dy,
    int& ier
) {
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
            std::cerr << "bcspeval: xget=" << xget << " out of range " << x[0] << " to " << x[nx - 1] << "\n";
        } else {
            if ((xget < x[0] - 0.5 * zxtol) || (xget > x[nx - 1] + 0.5 * zxtol)) {
                std::cerr << "bcspeval: xget=" << xget << " beyond range " << x[0] << " to " << x[nx - 1] << " (fixup applied)\n";
            }
            zxget = (xget < x[0]) ? x[0] : x[nx - 1];
        }
    }

    if ((yget < y[0]) || (yget > y[ny - 1])) {
        double zytol = 4.0e-7 * std::max(std::abs(y[0]), std::abs(y[ny - 1]));
        if ((yget < y[0] - zytol) || (yget > y[ny - 1] + zytol)) {
            ier = 1;
            std::cerr << "bcspeval: yget=" << yget << " out of range " << y[0] << " to " << y[ny - 1] << "\n";
        } else {
            if ((yget < y[0] - 0.5 * zytol) || (yget > y[ny - 1] + 0.5 * zytol)) {
                std::cerr << "bcspeval: yget=" << yget << " beyond range " << y[0] << " to " << y[ny - 1] << " (fixup applied)\n";
            }
            zyget = (yget < y[0]) ? y[0] : y[ny - 1];
        }
    }

    if (ier != 0) return;

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
void BiCubicSplineInterpolator<T, MemorySpace>::bcspevfn(
    const std::vector<int>& ict,             // Selector array for which derivatives to compute
    int ivec,                     // Number of vector points
    int ivd,                      // First dimension of fval (≥ ivec)
    Rank2View<T, MemorySpace> fval,                 // Output array: size [ivd, *] (flattened)
    std::vector<int>& iv,                // Grid cell indices in x direction
    std::vector<int>& jv,                // Grid cell indices in y direction
    std::vector<double>& dxv,            // x displacements within cells
    std::vector<double>& dyv,            // y displacements within cells
    Rank4View<T, MemorySpace> f,              // Bicubic spline coefficients [4, 4, inf3, ny] (flattened)
    int inf3,                     // 3rd dimension of f
    int ny                        // Number of y points (4th dim of f)
){
    int iaval = 0; // Index for fval
    if (ict[0] <= 2) {
        if ((ict[0] > 0) || (ict[0] == -1)) {
            // Evaluate f
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                int j = jv[v];
                double dx = dxv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    f(0, 0, i, j) + dy * (f(0, 1, i, j) + dy * (f(0, 2, i, j) + dy * f(0, 3, i, j))) +
                    dx * (f(1, 0, i, j) + dy * (f(1, 1, i, j) + dy * (f(1, 2, i, j) + dy * f(1, 3, i, j))) +
                    dx * (f(2, 0, i, j) + dy * (f(2, 1, i, j) + dy * (f(2, 2, i, j) + dy * f(2, 3, i, j))) +
                    dx * (f(3, 0, i, j) + dy * (f(3, 1, i, j) + dy * (f(3, 2, i, j) + dy * f(3, 3, i, j))))));
            }
        }

        if ((ict[1] > 0) && (ict[0] != -1)) {
            // Evaluate df/dx
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                int j = jv[v];
                double dx = dxv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    f(1, 0, i, j) + dy * (f(1, 1, i, j) + dy * (f(1, 2, i, j) + dy * f(1, 3, i, j))) +
                    2.0 * dx * (
                        f(2, 0, i, j) + dy * (f(2, 1, i, j) + dy * (f(2, 2, i, j) + dy * f(2, 3, i, j))) +
                        1.5 * dx * (
                            f(3, 0, i, j) + dy * (f(3, 1, i, j) + dy * (f(3, 2, i, j) + dy * f(3, 3, i, j)))));
            }
        }

        if ((ict[2] > 0) && (ict[0] != -1)) {
            // Evaluate df/dy
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                int j = jv[v];
                double dx = dxv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    f(0, 1, i, j) + dy * (2.0 * f(0, 2, i, j) + dy * 3.0 * f(0, 3, i, j)) +
                    dx * (f(1, 1, i, j) + dy * (2.0 * f(1, 2, i, j) + dy * 3.0 * f(1, 3, i, j)) +
                    dx * (f(2, 1, i, j) + dy * (2.0 * f(2, 2, i, j) + dy * 3.0 * f(2, 3, i, j)) +
                    dx * (f(3, 1, i, j) + dy * (2.0 * f(3, 2, i, j) + dy * 3.0 * f(3, 3, i, j)))));
            }
        }

        if ((ict[3] > 0) || (ict[0] == -1)) {
            // Evaluate d2f/dx2
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                int j = jv[v];
                double dx = dxv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    2.0 * (f(2, 0, i, j) + dy * (f(2, 1, i, j) + dy * (f(2, 2, i, j) + dy * f(2, 3, i, j)))) +
                    6.0 * dx * (f(3, 0, i, j) + dy * (f(3, 1, i, j) + dy * (f(3, 2, i, j) + dy * f(3, 3, i, j))));
            }
        }

        if ((ict[4] > 0) || (ict[0] == -1)) {
            // Evaluate d2f/dy2
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                int j = jv[v];
                double dx = dxv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    2.0 * f(0, 2, i, j) + 6.0 * dy * f(0, 3, i, j) +
                    dx * (2.0 * f(1, 2, i, j) + 6.0 * dy * f(1, 3, i, j) +
                    dx * (2.0 * f(2, 2, i, j) + 6.0 * dy * f(2, 3, i, j) +
                    dx * (2.0 * f(3, 2, i, j) + 6.0 * dy * f(3, 3, i, j))));
            }
        }

        if ((ict[5] > 0) && (ict[0] != -1)) {
            // Evaluate d2f/dxdy
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                int j = jv[v];
                double dx = dxv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    f(1, 1, i, j) + dy * (2.0 * f(1, 2, i, j) + dy * 3.0 * f(1, 3, i, j)) +
                    2.0 * dx * (f(2, 1, i, j) + dy * (2.0 * f(2, 2, i, j) + dy * 3.0 * f(2, 3, i, j)) +
                    1.5 * dx * (f(3, 1, i, j) + dy * (2.0 * f(3, 2, i, j) + dy * 3.0 * f(3, 3, i, j))));
            }
        }

        if (ict[0] == -1) {
            // Evaluate d4f/dx2dy2
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v];
                int j = jv[v];
                double dx = dxv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    4.0 * f(2, 2, i, j) + 12.0 * dy * f(2, 3, i, j) +
                    dx * (12.0 * f(3, 2, i, j) + 36.0 * dy * f(3, 3, i, j));
            }
        }
    } else if (ict[0] == 3) {
        if (ict[1] == 1) {
            // d³f/dx³ (not continuous)
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    6.0 * (f(3, 0, i, j) + dy * (f(3, 1, i, j) + dy * (f(3, 2, i, j) + dy * f(3, 3, i, j))));
            }
        }

        if (ict[2] == 1) {
            // d³f/dx²dy
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dx = dxv[v], dy = dyv[v];
                fval(v, iaval - 1) =
                    2.0 * (f(2, 1, i, j) + dy * (2.0 * f(2, 2, i, j) + dy * 3.0 * f(2, 3, i, j))) +
                    6.0 * dx * (f(3, 1, i, j) + dy * (2.0 * f(3, 2, i, j) + dy * 3.0 * f(3, 3, i, j)));
            }
        }

        if (ict[3] == 1) {
            // d³f/dxdy²
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dx = dxv[v], dy = dyv[v];
                fval(v, iaval - 1) =
                    2.0 * f(1, 2, i, j) + 6.0 * dy * f(1, 3, i, j) +
                    2.0 * dx * (
                        2.0 * f(2, 2, i, j) + 6.0 * dy * f(2, 3, i, j) +
                        1.5 * dx * (2.0 * f(3, 2, i, j) + 6.0 * dy * f(3, 3, i, j)));
            }
        }

        if (ict[4] == 1) {
            // d³f/dy³ (not continuous)
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dx = dxv[v];
                fval(v, iaval - 1) =
                    6.0 * (f(0, 3, i, j) + dx * (f(1, 3, i, j) + dx * (f(2, 3, i, j) + dx * f(3, 3, i, j))));
            }
        }

    } else if (ict[0] == 4) {
        if (ict[1] == 1) {
            // d⁴f/dx³dy
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    6.0 * (f(3, 1, i, j) + dy * 2.0 * (f(3, 2, i, j) + dy * 1.5 * f(3, 3, i, j)));
            }
        }

        if (ict[2] == 1) {
            // d⁴f/dx²dy²
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dx = dxv[v], dy = dyv[v];
                fval(v, iaval - 1) =
                    4.0 * f(2, 2, i, j) + 12.0 * dy * f(2, 3, i, j) +
                    dx * (12.0 * f(3, 2, i, j) + 36.0 * dy * f(3, 3, i, j));
            }
        }

        if (ict[3] == 1) {
            // d⁴f/dxdy³ (not continuous)
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dx = dxv[v];
                fval(v, iaval - 1) =
                    6.0 * (f(1, 3, i, j) + 2.0 * dx * (f(2, 3, i, j) + 1.5 * dx * f(3, 3, i, j)));
            }
        }

    } else if (ict[0] == 5) {
        if (ict[1] == 1) {
            // d⁵f/dx³dy² (not continuous)
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dy = dyv[v];
                fval(v, iaval - 1) =
                    12.0 * (f(3, 2, i, j) + dy * 3.0 * f(3, 3, i, j));
            }
        }

        if (ict[2] == 1) {
            // d⁵f/dx²dy³ (not continuous)
            iaval++;
            for (int v = 0; v < ivec; ++v) {
                int i = iv[v], j = jv[v];
                double dx = dxv[v];
                fval(v, iaval - 1) =
                    12.0 * (f(2, 3, i, j) + dx * 3.0 * f(3, 3, i, j));
            }
        }

    } else if (ict[0] == 6) {
        // d⁶f/dx³dy³ (not continuous)
        iaval++;
        for (int v = 0; v < ivec; ++v) {
            int i = iv[v], j = jv[v];
            fval(v, iaval - 1) = 36.0 * f(3, 3, i, j);
        }
    }
}// end bcspevfn


template <typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::mkbicub(
    Rank1View<const T, MemorySpace> x, int nx,
    Rank1View<const T, MemorySpace> y, int ny,
    Rank3View<T, MemorySpace> f, int nf2,
    int ibcxmin, Rank1View<T, MemorySpace> bcxmin,
    int ibcxmax, Rank1View<T, MemorySpace> bcxmax,
    int ibcymin, Rank1View<T, MemorySpace> bcymin,
    int ibcymax, Rank1View<T, MemorySpace> bcymax,
    int& ilinx, int& iliny, int& ier
) {
    ier = 0;
    int iflg2 = 0;

    // Check if inhomogeneous y-boundary conditions exist
    if (ibcymin != -1) {
        if ((ibcymin == 1 || ibcymin == 2)) {
            for (int ix = 0; ix < nx; ++ix) {
                if (bcymin[ix] != 0.0) iflg2 = 1;
            }
        }
        if ((ibcymax == 1 || ibcymax == 2)) {
            for (int ix = 0; ix < nx; ++ix) {
                if (bcymax[ix] != 0.0) iflg2 = 1;
            }
        }
    }

    // Check bc validity (implement ibc_ck separately)
    ibc_ck(ibcxmin, "bcspline", "xmin", -1, 7, ier);
    if (ibcxmin >= 0) ibc_ck(ibcxmax, "bcspline", "xmax", 0, 7, ier);
    ibc_ck(ibcymin, "bcspline", "ymin", -1, 7, ier);
    if (ibcymin >= 0) ibc_ck(ibcymax, "bcspline", "ymax", 0, 7, ier);

    // Check if x and y are strictly ascending (implement splinck separately)
    CubicSplineInterpolator<T, MemorySpace>::splinck(x, 1.0e-3);
    CubicSplineInterpolator<T, MemorySpace>::splinck(y, 1.0e-3);

    std::vector<T> fwk_x_vec(2 * nx);
    auto fwk_x = Rank2View<T, MemorySpace>(fwk_x_vec.data(), 2, nx);
    std::vector<T> fwk4_x_vec(4 * nx);
    auto fwk4_x = Rank2View<T, MemorySpace>(fwk4_x_vec.data(), 4, nx);
    std::vector<T> fwk_y_vec(2 * ny);
    auto fwk_y = Rank2View<T, MemorySpace>(fwk_y_vec.data(), 2, ny);
    std::vector<T> fwk4_y_vec(4 * ny);
    auto fwk4_y = Rank2View<T, MemorySpace>(fwk4_y_vec.data(), 4, ny);
    std::vector<T> wk_x_vec(nx);
    auto wk_x = Rank1View<T, MemorySpace>(wk_x_vec.data(), nx);
    std::vector<T> wk_y_vec(ny);
    auto wk_y = Rank1View<T, MemorySpace>(wk_y_vec.data(), ny);

    double zbcmin = 0.0, zbcmax = 0.0;

    // Compute fxx
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix)
            fwk_x(0, ix) = f(0, ix, iy);

        if (ibcxmin == 1 || ibcxmin == 2) zbcmin = bcxmin[iy];
        if (ibcxmax == 1 || ibcxmax == 2) zbcmax = bcxmax[iy];

        CubicSplineInterpolator<T, MemorySpace>::mkspline(x, nx, fwk_x, fwk4_x, ibcxmin, zbcmin, ibcxmax, zbcmax, wk_x);
        if (ier != 0) return;

        for (int ix = 0; ix < nx; ++ix)
            f(1, ix, iy) = fwk_x(1, ix);
    }

    zbcmin = 0.0;
    zbcmax = 0.0;
    int ibcmin = ibcymin, ibcmax = ibcymax;
    // Compute fyy
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy)
            fwk_y(0, iy) = f(0, ix, iy);

        if (iflg2 == 1) {
            if (ibcymin == 1 || ibcymin == 2) ibcmin = 0;
            if (ibcymax == 1 || ibcymax == 2) ibcmax = 0;
        }

        CubicSplineInterpolator<T, MemorySpace>::mkspline(y, ny, fwk_y, fwk4_y, ibcmin, 0.0, ibcmax, 0.0, wk_y);
        if (ier != 0) return;

        for (int iy = 0; iy < ny; ++iy)
            f(2, ix, iy) = fwk_y(1, iy);
    }

    zbcmin = 0.0;
    zbcmax = 0.0;
    ibcmin = ibcymin;
    ibcmax = ibcymax;
    // Compute fxxyy
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy)
            fwk_y(0, iy) = f(1, ix, iy);

        if (iflg2 == 1) {
            if (ibcymin == 1 || ibcymin == 2) ibcmin = 0;
            if (ibcymax == 1 || ibcymax == 2) ibcmax = 0;
        }

        CubicSplineInterpolator<T, MemorySpace>::mkspline(y, ny, fwk_y, fwk4_y, ibcmin, 0.0, ibcmax, 0.0, wk_y);
        if (ier != 0) return;

        for (int iy = 0; iy < ny; ++iy)
            f(3, ix, iy) = fwk_y(1, iy);
    }

    // Correct for inhomogeneous boundary conditions if needed
    if (iflg2 == 1) {
        std::vector<std::vector<std::vector<T>>> fcorr(2, std::vector<std::vector<T>>(nx, std::vector<T>(ny, 0.0)));
        std::vector<T> zdiff(2, 0.0);

        for (int ix = 0; ix < nx; ++ix) {
            zdiff[0] = (ibcymin == 1) ? (bcymin[ix] - ((f(0, ix, 1) - f(0, ix, 0)) / (y[1] - y[0]) + (y[1] - y[0]) * (-2.0 * f(2, ix, 0) - f(2, ix, 1)) / 6.0)) :
                        ((ibcymin == 2) ? bcymin[ix] - f(2, ix, 0) : 0.0);

            zdiff[1] = (ibcymax == 1) ? (bcymax[ix] - ((f(0, ix, ny-1) - f(0, ix, ny-2)) / (y[ny-1] - y[ny-2]) + (y[ny-1] - y[ny-2]) * (2.0 * f(2, ix, ny-1) + f(2, ix, ny-2)) / 6.0)) :
                        ((ibcymax == 2) ? bcymax[ix] - f(2, ix, ny-1) : 0.0);

            for (int iy = 0; iy < ny; ++iy) {
                fwk_y(0, iy) = 0.0;
            }
            CubicSplineInterpolator<T, MemorySpace>::mkspline(y, ny, fwk_y, fwk4_y, ibcymin, zdiff[0], ibcymax, zdiff[1], wk_y);
            if (ier != 0) return;
            for (int iy = 0; iy < ny; ++iy)
                fcorr[0][ix][iy] = fwk_y(1, iy);
        }

        zbcmin=0;
        zbcmax=0;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix)
                fwk_x(0, ix) = fcorr[0][ix][iy];
            CubicSplineInterpolator<T, MemorySpace>::mkspline(x, nx, fwk_x, fwk4_x, ibcxmin, 0.0, ibcxmax, 0.0, wk_x);
            if (ier != 0) return;
            for (int ix = 0; ix < nx; ++ix)
                fcorr[1][ix][iy] = fwk_x(1, ix);
        }

        for (int c = 0; c < 2; ++c)
            for (int ix = 0; ix < nx; ++ix)
                for (int iy = 0; iy < ny; ++iy)
                    f(c + 2, ix, iy) += fcorr[c][ix][iy];
    }
}

template<typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::herm2xy(double xget, double yget,
                    Rank1View<const T, MemorySpace> x,
                    int& nx,
                    Rank1View<const T, MemorySpace> y,
                    int& ny,
                    bool ilinx, bool iliny,
                    int& i, int& j,
                    double& xparam, double& yparam,
                    double& hx, double& hxi,
                    double& hy, double& hyi,
                    int& ier)
{
    ier = 0;

    if (nx < 2 || ny < 2)
    {
        ier = 1;
        std::cerr << "?herm2xy: grid must have at least two points in each direction\n";
        return;
    }

    double zxget = xget;
    double zyget = yget;

    const double zxtol = 4.0e-7 * std::max(std::fabs(x[0]), std::fabs(x[nx - 1]));
    const double zytol = 4.0e-7 * std::max(std::fabs(y[0]), std::fabs(y[ny - 1]));

    // X range check / fixup
    if (xget < x[0] - zxtol || xget > x[nx - 1] + zxtol)
    {
        ier = 1;
        std::cerr << "herm2xy:  xget = " << xget
                  << " out of range " << x[0] << " to " << x[nx - 1] << '\n';
        return;
    }
    else if (xget < x[0])
    {
        if (xget < x[0] - 0.5 * zxtol || xget > x[nx - 1] + 0.5 * zxtol)
            std::cerr << "%herm2xy:  xget = " << xget
                      << " beyond range " << x[0] << " to " << x[nx - 1]
                      << " (fixup applied)\n";
        zxget = x[0];
    }
    else if (xget > x[nx - 1])
    {
        if (xget < x[0] - 0.5 * zxtol || xget > x[nx - 1] + 0.5 * zxtol)
            std::cerr << "%herm2xy:  xget = " << xget
                      << " beyond range " << x[0] << " to " << x[nx - 1]
                      << " (fixup applied)\n";
        zxget = x[nx - 1];
    }

    // Y range check / fixup
    if (yget < y[0] - zytol || yget > y[ny - 1] + zytol)
    {
        ier = 1;
        std::cerr << "?herm2xy:  yget = " << yget
                  << " out of range " << y[0] << " to " << y[ny - 1] << '\n';
        return;
    }
    else if (yget < y[0])
    {
        if (yget < y[0] - 0.5 * zytol || yget > y[ny - 1] + 0.5 * zytol)
            std::cerr << "%herm2xy:  yget = " << yget
                      << " beyond range " << y[0] << " to " << y[ny - 1]
                      << " (fixup applied)\n";
        zyget = y[0];
    }
    else if (yget > y[ny - 1])
    {
        if (yget < y[0] - 0.5 * zytol || yget > y[ny - 1] + 0.5 * zytol)
            std::cerr << "%herm2xy:  yget = " << yget
                      << " beyond range " << y[0] << " to " << y[ny - 1]
                      << " (fixup applied)\n";
        zyget = y[ny - 1];
    }


    int nxm = nx - 1;  // Number of intervals in x
    int nym = ny - 1;  // Number of intervals in y
    // Determine zone index i
    int ii = static_cast<int>(nxm * (zxget - x[0]) / (x[nx - 1] - x[0]));
    i = std::min(nxm - 1, ii);
    if (zxget < x[i]) --i;
    else if (zxget > x[i + 1]) ++i;
    
    
    // Determine zone index j
    ii = static_cast<int>(nym * (zyget - y[0]) / (y[ny - 1] - y[0]));
    j = std::min(nym - 1, ii);
    if (zyget < y[j]) --j;
    else if (zyget > y[j + 1]) ++j;

    hx = x[i + 1] - x[i];
    hy = y[j + 1] - y[j];

    hxi = 1.0 / hx;
    hyi = 1.0 / hy;

    xparam = (zxget - x[i]) * hxi;
    yparam = (zyget - y[j]) * hyi;
}

template<typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::fvbicub(const std::vector<int>& ict, int ivec, int ivecd,
                    Rank2View<T, MemorySpace> fval,
                    const std::vector<int>& ii,
                    const std::vector<int>& jj,
                    const std::vector<double>& xparam,
                    const std::vector<double>& yparam,
                    const std::vector<double>& hx,
                    const std::vector<double>& hxi,
                    const std::vector<double>& hy,
                    const std::vector<double>& hyi,
                    Rank3View<T, MemorySpace> fin)
{
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
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double xp2 = xp * xp, xpi2 = xpi * xpi;
                double cx  = xp  * (xp2  - 1.0);
                double cxi = xpi * (xpi2 - 1.0);
                double hx2 = hx[v] * hx[v];

                double yp = yparam[v], ypi = 1.0 - yp;
                double yp2 = yp * yp, ypi2 = ypi * ypi;
                double cy  = yp  * (yp2  - 1.0);
                double cyi = ypi * (ypi2 - 1.0);
                double hy2 = hy[v] * hy[v];

                double sum = xpi * (ypi * fin(0, i, j)     + yp * fin(0, i, j + 1)) +
                             xp  * (ypi * fin(0, i + 1, j) + yp * fin(0, i + 1, j + 1));
                sum += sixth * hx2 * (
                           cxi * (ypi * fin(1, i, j)     + yp * fin(1, i, j + 1)) +
                           cx  * (ypi * fin(1, i + 1, j) + yp * fin(1, i + 1, j + 1)));
                sum += sixth * hy2 * (
                           xpi * (cyi * fin(2, i, j)     + cy * fin(2, i, j + 1)) +
                           xp  * (cyi * fin(2, i + 1, j) + cy * fin(2, i + 1, j + 1)));
                sum += z36th * hx2 * hy2 * (
                           cxi * (cyi * fin(3, i, j)     + cy * fin(3, i, j + 1)) +
                           cx  * (cyi * fin(3, i + 1, j) + cy * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** df/dx **********/
        if (ict[1] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double xp2 = xp * xp, xpi2 = xpi * xpi;
                double cxd  = 3.0 * xp2 - 1.0;
                double cxdi = -3.0 * xpi2 + 1.0;
                double yp = yparam[v], ypi = 1.0 - yp;
                double yp2 = yp * yp, ypi2 = ypi * ypi;
                double cy  = yp  * (yp2  - 1.0);
                double cyi = ypi * (ypi2 - 1.0);
                double hy2 = hy[v] * hy[v];

                double sum = hxi[v] * (-(ypi * fin(0, i, j)     + yp * fin(0, i, j + 1)) +
                                        (ypi * fin(0, i + 1, j) + yp * fin(0, i + 1, j + 1)));
                sum += sixth * hx[v] * (
                           cxdi * (ypi * fin(1, i, j)     + yp * fin(1, i, j + 1)) +
                           cxd  * (ypi * fin(1, i + 1, j) + yp * fin(1, i + 1, j + 1)));
                sum += sixth * hxi[v] * hy2 * (
                           -(cyi * fin(2, i, j)     + cy * fin(2, i, j + 1)) +
                             (cyi * fin(2, i + 1, j) + cy * fin(2, i + 1, j + 1)));
                sum += z36th * hx[v] * hy2 * (
                           cxdi * (cyi * fin(3, i, j)     + cy * fin(3, i, j + 1)) +
                           cxd  * (cyi * fin(3, i + 1, j) + cy * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** df/dy **********/
        if (ict[2] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double xp2 = xp * xp, xpi2 = xpi * xpi;
                double cx  = xp  * (xp2  - 1.0);
                double cxi = xpi * (xpi2 - 1.0);
                double hx2 = hx[v] * hx[v];
                double yp = yparam[v], ypi = 1.0 - yp;
                double yp2 = yp * yp, ypi2 = ypi * ypi;
                double cyd  = 3.0 * yp2 - 1.0;
                double cydi = -3.0 * ypi2 + 1.0;

                double sum = hyi[v] * (xpi * (-fin(0, i, j)     + fin(0, i, j + 1)) +
                                        xp  * (-fin(0, i + 1, j) + fin(0, i + 1, j + 1)));
                sum += sixth * hx2 * hyi[v] * (
                           cxi * (-fin(1, i, j)     + fin(1, i, j + 1)) +
                           cx  * (-fin(1, i + 1, j) + fin(1, i + 1, j + 1)));
                sum += sixth * hy[v] * (
                           xpi * (cydi * fin(2, i, j)     + cyd * fin(2, i, j + 1)) +
                           xp  * (cydi * fin(2, i + 1, j) + cyd * fin(2, i + 1, j + 1)));
                sum += z36th * hx2 * hy[v] * (
                           cxi * (cydi * fin(3, i, j)     + cyd * fin(3, i, j + 1)) +
                           cx  * (cydi * fin(3, i + 1, j) + cyd * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d²f/dx² **********/
        if (ict[3] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double yp = yparam[v], ypi = 1.0 - yp;
                double yp2 = yp * yp, ypi2 = ypi * ypi;
                double cy  = yp  * (yp2  - 1.0);
                double cyi = ypi * (ypi2 - 1.0);
                double hy2 = hy[v] * hy[v];

                double sum = xpi * (ypi * fin(1, i, j)     + yp * fin(1, i, j + 1)) +
                             xp  * (ypi * fin(1, i + 1, j) + yp * fin(1, i + 1, j + 1));
                sum += sixth * hy2 * (
                           xpi * (cyi * fin(3, i, j)     + cy * fin(3, i, j + 1)) +
                           xp  * (cyi * fin(3, i + 1, j) + cy * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d²f/dy² **********/
        if (ict[4] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double xp2 = xp * xp, xpi2 = xpi * xpi;
                double cx  = xp  * (xp2  - 1.0);
                double cxi = xpi * (xpi2 - 1.0);
                double hx2 = hx[v] * hx[v];
                double yp = yparam[v], ypi = 1.0 - yp;

                double sum = xpi * (ypi * fin(2, i, j)     + yp * fin(2, i, j + 1)) +
                             xp  * (ypi * fin(2, i + 1, j) + yp * fin(2, i + 1, j + 1));
                sum += sixth * hx2 * (
                           cxi * (ypi * fin(3, i, j)     + yp * fin(3, i, j + 1)) +
                           cx  * (ypi * fin(3, i + 1, j) + yp * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d²f/dxdy **********/
        if (ict[5] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double xp2 = xp * xp, xpi2 = xpi * xpi;
                double cxd  = 3.0 * xp2 - 1.0;
                double cxdi = -3.0 * xpi2 + 1.0;
                double yp = yparam[v], ypi = 1.0 - yp;
                double yp2 = yp * yp, ypi2 = ypi * ypi;
                double cyd  = 3.0 * yp2 - 1.0;
                double cydi = -3.0 * ypi2 + 1.0;

                double sum = hxi[v] * hyi[v] * (fin(0, i, j) - fin(0, i, j + 1) -
                                                fin(0, i + 1, j) + fin(0, i + 1, j + 1));
                sum += sixth * hx[v] * hyi[v] * (
                           cxdi * (-fin(1, i, j)     + fin(1, i, j + 1)) +
                           cxd  * (-fin(1, i + 1, j) + fin(1, i + 1, j + 1)));
                sum += sixth * hxi[v] * hy[v] * (
                           -(cydi * fin(2, i, j)     + cyd * fin(2, i, j + 1)) +
                             (cydi * fin(2, i + 1, j) + cyd * fin(2, i + 1, j + 1)));
                sum += z36th * hx[v] * hy[v] * (
                           cxdi * (cydi * fin(3, i, j)     + cyd * fin(3, i, j + 1)) +
                           cxd  * (cydi * fin(3, i + 1, j) + cyd * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }
    }

    /* ------------------------------------------------------------------
     * ict[0] = 3  → 3rd‑order derivative combinations
     * ----------------------------------------------------------------*/
    else if (ict[0] == 3) {
        /********** d³f/dx³ **********/
        if (ict[1] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double yp = yparam[v], ypi = 1.0 - yp;
                double yp2 = yp * yp, ypi2 = ypi * ypi;
                double cy  = yp  * (yp2  - 1.0);
                double cyi = ypi * (ypi2 - 1.0);
                double hy2 = hy[v] * hy[v];
                double sum = hxi[v] * (-(ypi * fin(1, i, j)     + yp * fin(1, i, j + 1)) +
                                        (ypi * fin(1, i + 1, j) + yp * fin(1, i + 1, j + 1)));
                sum += sixth * hy2 * hxi[v] * (-(cyi * fin(3, i, j)     + cy * fin(3, i, j + 1)) +
                                                (cyi * fin(3, i + 1, j) + cy * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d³f/dx²dy **********/
        if (ict[2] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double yp = yparam[v], ypi = 1.0 - yp;
                double yp2 = yp * yp, ypi2 = ypi * ypi;
                double cyd  = 3.0 * yp2 - 1.0;
                double cydi = -3.0 * ypi2 + 1.0;
                double sum = hyi[v] * (xpi * (-fin(1, i, j)     + fin(1, i, j + 1)) +
                                        xp  * (-fin(1, i + 1, j) + fin(1, i + 1, j + 1)));
                sum += sixth * hy[v] * (xpi * (cydi * fin(3, i, j)     + cyd * fin(3, i, j + 1)) +
                                        xp  * (cydi * fin(3, i + 1, j) + cyd * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d³f/dxdy² **********/
        if (ict[3] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double xp2 = xp * xp, xpi2 = xpi * xpi;
                double cxd  = 3.0 * xp2 - 1.0;
                double cxdi = -3.0 * xpi2 + 1.0;
                double yp = yparam[v], ypi = 1.0 - yp;
                double sum = hxi[v] * (-(ypi * fin(2, i, j)     + yp * fin(2, i, j + 1)) +
                                        (ypi * fin(2, i + 1, j) + yp * fin(2, i + 1, j + 1)));
                sum += sixth * hx[v] * (
                           cxdi * (ypi * fin(3, i, j)     + yp * fin(3, i, j + 1)) +
                           cxd  * (ypi * fin(3, i + 1, j) + yp * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d³f/dy³ **********/
        if (ict[4] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double xp2 = xp * xp, xpi2 = xpi * xpi;
                double cx  = xp  * (xp2  - 1.0);
                double cxi = xpi * (xpi2 - 1.0);
                double hx2 = hx[v] * hx[v];
                double sum = hyi[v] * (xpi * (-fin(2, i, j)     + fin(2, i, j + 1)) +
                                        xp  * (-fin(2, i + 1, j) + fin(2, i + 1, j + 1)));
                sum += sixth * hx2 * hyi[v] * (
                           cxi * (-fin(3, i, j)     + fin(3, i, j + 1)) +
                           cx  * (-fin(3, i + 1, j) + fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }
    }

    /* ------------------------------------------------------------------
     * ict[0] = 4  → 4th‑order derivative combinations
     * ----------------------------------------------------------------*/
    else if (ict[0] == 4) {
        /********** d⁴f/dx³dy **********/
        if (ict[1] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double yp = yparam[v], ypi = 1.0 - yp;
                double yp2 = yp * yp, ypi2 = ypi * ypi;
                double cyd  = 3.0 * yp2 - 1.0;
                double cydi = -3.0 * ypi2 + 1.0;
                double sum = hxi[v] * hyi[v] * (fin(1, i, j) - fin(1, i, j + 1) -
                                                  fin(1, i + 1, j) + fin(1, i + 1, j + 1));
                sum += sixth * hy[v] * hxi[v] * (-(cydi * fin(3, i, j)     + cyd * fin(3, i, j + 1)) +
                                                  (cydi * fin(3, i + 1, j) + cyd * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d⁴f/dx²dy² **********/
        if (ict[2] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double yp = yparam[v], ypi = 1.0 - yp;
                double sum = xpi * (ypi * fin(3, i, j)     + yp * fin(3, i, j + 1)) +
                             xp  * (ypi * fin(3, i + 1, j) + yp * fin(3, i + 1, j + 1));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d⁴f/dxdy³ **********/
        if (ict[3] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double xp2 = xp * xp, xpi2 = xpi * xpi;
                double cxd  = 3.0 * xp2 - 1.0;
                double cxdi = -3.0 * xpi2 + 1.0;
                double sum = hxi[v] * hyi[v] * (fin(2, i, j) - fin(2, i, j + 1) -
                                                  fin(2, i + 1, j) + fin(2, i + 1, j + 1));
                sum += sixth * hx[v] * hyi[v] * (
                           cxdi * (-fin(3, i, j)     + fin(3, i, j + 1)) +
                           cxd  * (-fin(3, i + 1, j) + fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }
    }

    /* ------------------------------------------------------------------
     * ict[0] = 5  → 5th‑order derivative combinations
     * ----------------------------------------------------------------*/
    else if (ict[0] == 5) {
        /********** d⁵f/dx³dy² **********/
        if (ict[1] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double yp = yparam[v], ypi = 1.0 - yp;
                double sum = hxi[v] * (-(ypi * fin(3, i, j)     + yp * fin(3, i, j + 1)) +
                                         (ypi * fin(3, i + 1, j) + yp * fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }

        /********** d⁵f/dx²dy³ **********/
        if (ict[2] == 1) {
            iadr++;
            for (int v = 0; v < ivec; ++v) {
                int i = ii[v], j = jj[v];
                double xp = xparam[v], xpi = 1.0 - xp;
                double sum = hyi[v] * (xpi * (-fin(3, i, j)     + fin(3, i, j + 1)) +
                                        xp  * (-fin(3, i + 1, j) + fin(3, i + 1, j + 1)));
                fval(v, iadr - 1) = sum;
            }
        }
    }

    /* ------------------------------------------------------------------
     * ict[0] = 6  → 6th‑order derivative (dx³dy³)
     * ----------------------------------------------------------------*/
    else if (ict[0] == 6) {
        iadr++;
        for (int v = 0; v < ivec; ++v) {
            int i = ii[v], j = jj[v];
            double sum = hxi[v] * hyi[v] * (fin(3, i, j) - fin(3, i, j + 1) -
                                            fin(3, i + 1, j) + fin(3, i + 1, j + 1));
            fval(v, iadr - 1) = sum;
        }
    }
}

template<typename T, typename MemorySpace>
void BiCubicSplineInterpolator<T, MemorySpace>::evbicub(double xget, double yget,
             Rank1View<const T, MemorySpace> x, int nx,
             Rank1View<const T, MemorySpace> y, int ny,
             int ilinx, int iliny,
             Rank3View<T, MemorySpace> f, // f[4][inf2][ny]
             int inf2,
             const std::vector<int>& ict,
             Rank2View<T, MemorySpace> fval, // output (size depends on ict)
             int& ier)
{
    // Local variables
    int i = 0, j = 0;
    T xparam = 0.0, yparam = 0.0;
    T hx = 0.0, hy = 0.0;
    T hxi = 0.0, hyi = 0.0;

    // Call herm2xy to locate cell and compute params
    herm2xy(xget, yget, x, nx, y, ny, ilinx, iliny,
            i, j, xparam, yparam, hx, hxi, hy, hyi, ier);

    if (ier != 0) return;

    // Evaluate spline and derivatives
    // Wrap scalar values into single-element vectors
    std::vector<int> ii{ i }, jj{ j };
    std::vector<T> xparams{ xparam }, yparams{ yparam };
    std::vector<T> hxs{ hx }, hxis{ hxi };
    std::vector<T> hys{ hy }, hyis{ hyi };

    // Call fvbicub with scalar-vector interface
    fvbicub(ict, 1, 1,
            fval, ii, jj, xparams, yparams,
            hxs, hxis, hys, hyis,
            f);
}


} //namespace pcms
#endif