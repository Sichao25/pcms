%module pcms_mesh
%{
#include "pcms/capi/mesh.h"
%}
%include <../external/flibhpc/include/mpi.i>
%include <stdint.i>
%include <typemaps.i>

struct PcmsInterpolatorOmega_hLibraryHandle
{
  void* lib_handle;
};
typedef struct PcmsInterpolatorOmega_hLibraryHandle PcmsInterpolatorOmega_hLibraryHandle;

struct PcmsInterpolatorOmega_hMeshHandle
{
  void* mesh_handle;
};
typedef struct PcmsInterpolatorOmega_hMeshHandle PcmsInterpolatorOmega_hMeshHandle;


PcmsInterpolatorOmega_hLibraryHandle pcms_create_omega_h_library();
PcmsInterpolatorOmega_hMeshHandle pcms_create_omega_h_mesh(const char* filename, PcmsInterpolatorOmega_hLibraryHandle oh_lib_handle);
void pcms_destroy_omega_h_mesh(PcmsInterpolatorOmega_hMeshHandle oh_mesh);
void pcms_destroy_omega_h_library(PcmsInterpolatorOmega_hLibraryHandle oh_lib_handle);
