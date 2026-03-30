#ifndef PCMS_TRANSFER_OMEGA_H_CONSERVATIVE_PROJECTION_H
#define PCMS_TRANSFER_OMEGA_H_CONSERVATIVE_PROJECTION_H

#include "pcms/field/function_space.h"
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/transfer/conservative_projection_solver.hpp"
#include "pcms/transfer/transfer_operator.hpp"
#include <memory>

namespace pcms
{

// Conservative Galerkin projection between Omega_h order-1 Lagrange spaces.
//
// Construction caches the source-target mesh intersections. Apply() assembles
// the projection system for the current source coefficients, solves it, and
// writes the projected scalar field into the target Field<Real>.
class OmegaHConservativeProjection : public TransferOperator<Real>
{
public:
  OmegaHConservativeProjection(const FunctionSpace& source_space,
                               const FunctionSpace& target_space);

  void Apply(const Field<Real>& source, Field<Real>& target) const override;

private:
  std::shared_ptr<const OmegaHLagrangeLayout> source_layout_;
  std::shared_ptr<const OmegaHLagrangeLayout> target_layout_;
  IntersectionResults intersections_;
};

} // namespace pcms

#endif // PCMS_TRANSFER_OMEGA_H_CONSERVATIVE_PROJECTION_H
