#ifndef PCMS_OMEGA_H_MESH_CAPI_H
#define PCMS_OMEGA_H_MESH_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

struct PcmsInterpolatorOmega_hLibraryHandle
{
  void* lib_handle;
};
typedef struct PcmsInterpolatorOmega_hLibraryHandle
  PcmsInterpolatorOmega_hLibraryHandle;

/**
 * @brief Holds void pointers of Omega_h Mesh object and the library
 * (communicator) handle
 */
struct PcmsInterpolatorOmega_hMeshHandle
{
  void* mesh_handle;
};

/**
 * @brief Typedef for PcmsInterpolatorOmega_hMeshHandle struct
 * @copydetails PcmsInterpolatorOmega_hMeshHandle
 * @see PcmsInterpolatorOmega_hMeshHandle
 */
typedef struct PcmsInterpolatorOmega_hMeshHandle
  PcmsInterpolatorOmega_hMeshHandle;

/**
 * @brief Create Omega_h library handle on MPI_COMM_SELF if MPI is enabled
 * @return PcmsInterpolatorOmega_hLibraryHandle Handle to the created Omega_h
 * library
 * @note Must be deleted using pcms_destroy_omega_h_library to avoid memory
 * leaks.
 * @see pcms_create_omega_h_mesh, pcms_destroy_omega_h_library
 */
PcmsInterpolatorOmega_hLibraryHandle pcms_create_omega_h_library();

/**
 * @brief Read Omega_h mesh from file
 * @param filename C-string of Omega_h mesh filename
 * @param oh_lib_handle Handle to the Omega_h library
  created using pcms_create_omega_h_library
 * @return PcmsInterpolatorOmega_hMeshHandle Handle to the read Omega_h mesh
 *
 * @details Reads an Omega_h mesh from the given filename and returns a handle
 * to it. This handle can be used to create interpolators that operate on the
 * mesh.
 * @note Must be deleted using pcms_destroy_omega_h_mesh to avoid memory leaks.
 * @see pcms_create_interpolator, pcms_create_omega_h_library,
 pcms_destroy_omega_h_mesh
 */
PcmsInterpolatorOmega_hMeshHandle pcms_create_omega_h_mesh(
  const char* filename, PcmsInterpolatorOmega_hLibraryHandle oh_lib_handle);

/**
 * @brief Release Omega_h mesh
 * @param oh_mesh_handle Handle to the Omega_h mesh to be released
 *
 * @details Releases the Omega_h mesh associated with the given handle to
 * free up resources.
 */
void pcms_destroy_omega_h_mesh(
  PcmsInterpolatorOmega_hMeshHandle oh_mesh_handle);

/**
 * @brief Delete Omega_h library
 * @param oh_lib_handle
 *
 * @see pcms_create_omega_h_library
 */
void pcms_destroy_omega_h_library(
  PcmsInterpolatorOmega_hLibraryHandle oh_lib_handle);

#ifdef __cplusplus
}
#endif

#endif // PCMS_OMEGA_H_MESH_CAPI_H
