#ifndef PCMS_UTILITY_OMEGA_H_ARRAY_UTILS_H
#define PCMS_UTILITY_OMEGA_H_ARRAY_UTILS_H

#include <Omega_h_array.hpp>

namespace pcms
{

template <typename View2D>
Omega_h::Reals flatten_to_omega_h_reals(const View2D& coords,
                                        const char* label = "flat_coords")
{
  const int n = static_cast<int>(coords.extent(0));
  const int dim = static_cast<int>(coords.extent(1));

  Omega_h::HostWrite<Omega_h::Real> flat(n * dim, label);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < dim; ++d)
      flat[i * dim + d] = coords(i, d);

  return Omega_h::Reals(flat);
}

} // namespace pcms

#endif // PCMS_UTILITY_OMEGA_H_ARRAY_UTILS_H
