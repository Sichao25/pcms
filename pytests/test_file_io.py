"""
Tests for Omega_h file I/O Python bindings.
"""

import os
import shutil
import gc
import tempfile

import pytest
import PyOmega_h as omega_h


class TestFileIO:
    """Tests for reading and writing meshes in various file formats."""

    @staticmethod
    def _build_mesh(world, dims, divisions):
        """Helper: build a box mesh from the given dimensions / divisions."""
        return omega_h.build_box(
            world, omega_h.Family.SIMPLEX, *dims, *divisions, False,
        )

    @staticmethod
    def _mesh_props(mesh):
        """Return (dim, nverts, nelems) for a mesh."""
        return mesh.dim(), mesh.nverts(), mesh.nelems()

    def test_binary_io(self, omega_h_lib, world):
        """Test binary file format I/O."""
        mesh = self._build_mesh(world, (1.0, 1.0, 1.0), (4, 4, 4))
        dim_orig, nverts_orig, nelems_orig = self._mesh_props(mesh)

        test_dir = tempfile.mkdtemp(prefix="pcms_test_binary_")
        try:
            binary_file = os.path.join(test_dir, "test_mesh.osh")
            omega_h.write_mesh_binary(binary_file, mesh)
            mesh_read = omega_h.read_mesh_binary(binary_file, omega_h_lib)
            assert mesh_read.dim() == dim_orig
            assert mesh_read.nverts() == nverts_orig
            assert mesh_read.nelems() == nelems_orig
            del mesh_read
            del mesh
            gc.collect()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_gmsh_io(self, omega_h_lib, world):
        """Test Gmsh file format I/O."""
        mesh = self._build_mesh(world, (2.0, 1.0, 0.0), (5, 3, 0))
        dim_orig, nverts_orig, nelems_orig = self._mesh_props(mesh)

        test_dir = tempfile.mkdtemp(prefix="pcms_test_gmsh_")
        try:
            gmsh_file = os.path.join(test_dir, "test_mesh.msh")
            omega_h.write_mesh_gmsh(gmsh_file, mesh)
            mesh_read = omega_h.read_mesh_gmsh(gmsh_file, world)
            assert mesh_read.dim() == dim_orig
            assert mesh_read.nverts() == nverts_orig
            assert mesh_read.nelems() == nelems_orig
            del mesh_read
            del mesh
            gc.collect()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


    def test_vtk_io(self, omega_h_lib, world):
        """Test VTK file format I/O (VTU — write-only, for visualization)."""
        mesh = self._build_mesh(world, (1.5, 1.0, 0.5), (3, 3, 2))

        test_dir = tempfile.mkdtemp(prefix="pcms_test_vtk_")
        try:
            vtu_file = os.path.join(test_dir, "test_mesh.vtu")
            omega_h.write_mesh_vtu(vtu_file, mesh, compress=False)
            assert os.path.exists(vtu_file)

            vtu_compressed = os.path.join(test_dir, "test_mesh_compressed.vtu")
            omega_h.write_mesh_vtu(vtu_compressed, mesh, compress=True)
            assert os.path.exists(vtu_compressed)

            del mesh
            gc.collect()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_meshb_io(self, omega_h_lib, world):
        """Test MESHB file format I/O."""
        if not hasattr(omega_h, "write_mesh_meshb"):
            pytest.skip("MESHB support not available (OMEGA_H_USE_LIBMESHB not enabled)")

        mesh = self._build_mesh(world, (1.0, 1.0, 1.0), (3, 3, 3))
        dim_orig, nverts_orig, nelems_orig = self._mesh_props(mesh)

        test_dir = tempfile.mkdtemp(prefix="pcms_test_meshb_")
        try:
            meshb_file = os.path.join(test_dir, "test_mesh.mesh")
            omega_h.write_mesh_meshb(mesh, meshb_file, version=2)
            mesh_read = omega_h.read_mesh_meshb(meshb_file, omega_h_lib)
            assert mesh_read.dim() == dim_orig
            assert mesh_read.nverts() == nverts_orig
            assert mesh_read.nelems() == nelems_orig
            del mesh_read
            del mesh
            gc.collect()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_exodus_io(self, omega_h_lib, world):
        """Test Exodus file format I/O."""
        if not hasattr(omega_h, "write_mesh_exodus"):
            pytest.skip("Exodus support not available (OMEGA_H_USE_SEACASEXODUS not enabled)")

        mesh = self._build_mesh(world, (1.0, 1.0, 1.0), (3, 3, 3))
        dim_orig, nverts_orig, nelems_orig = self._mesh_props(mesh)

        test_dir = tempfile.mkdtemp(prefix="pcms_test_exodus_")
        try:
            exodus_file = os.path.join(test_dir, "test_mesh.exo")
            omega_h.write_mesh_exodus(exodus_file, mesh, verbose=False)
            if hasattr(omega_h, "exodus_open"):
                exo_handle = omega_h.exodus_open(exodus_file, verbose=False)
                mesh_from_handle = omega_h.OmegaHMesh(omega_h_lib)
                mesh_from_handle.set_comm(world)
                omega_h.read_mesh_exodus(exo_handle, mesh_from_handle, verbose=False)
                assert mesh_from_handle.dim() == dim_orig
                assert mesh_from_handle.nverts() == nverts_orig
                assert mesh_from_handle.nelems() == nelems_orig
                omega_h.exodus_close(exo_handle)
                del mesh_from_handle
            del mesh
            gc.collect()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_adios2_io(self, omega_h_lib, world):
        """Test ADIOS2 file format I/O."""
        if not hasattr(omega_h, "write_mesh_adios2"):
            pytest.skip("ADIOS2 support not available (OMEGA_H_USE_ADIOS2 not enabled)")

        mesh = self._build_mesh(world, (1.0, 1.0, 1.0), (3, 3, 3))
        dim_orig, nverts_orig, nelems_orig = self._mesh_props(mesh)

        test_dir = tempfile.mkdtemp(prefix="pcms_test_adios2_")
        try:
            adios2_file = os.path.join(test_dir, "test_mesh.bp")
            omega_h.write_mesh_adios2(adios2_file, mesh, prefix="")
            mesh_read = omega_h.read_mesh_adios2(adios2_file, omega_h_lib, prefix="")
            assert mesh_read.dim() == dim_orig
            assert mesh_read.nverts() == nverts_orig
            assert mesh_read.nelems() == nelems_orig
            del mesh_read
            del mesh
            gc.collect()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_read_mesh_file_auto_detect(self, omega_h_lib, world):
        """Test automatic format detection with read_mesh_file."""
        mesh = self._build_mesh(world, (1.0, 1.0, 1.0), (3, 3, 3))
        nverts_orig, nelems_orig = mesh.nverts(), mesh.nelems()

        test_dir = tempfile.mkdtemp(prefix="pcms_test_autodetect_")
        try:
            binary_file = os.path.join(test_dir, "mesh.osh")
            gmsh_file = os.path.join(test_dir, "mesh.msh")
            omega_h.write_mesh_binary(binary_file, mesh)
            omega_h.write_mesh_gmsh(gmsh_file, mesh)

            mesh_binary = omega_h.read_mesh_file(binary_file, world)
            assert mesh_binary.nverts() == nverts_orig
            assert mesh_binary.nelems() == nelems_orig

            mesh_gmsh = omega_h.read_mesh_file(gmsh_file, world)
            assert mesh_gmsh.nverts() == nverts_orig
            assert mesh_gmsh.nelems() == nelems_orig

            del mesh_binary
            del mesh_gmsh
            del mesh
            gc.collect()
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

