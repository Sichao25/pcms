"""
Field-centric tests for Omega_h-backed function spaces.
"""

import numpy as np
import pytest
import pcms
import PyOmega_h as omega_h

class TestOmegaHField:
    """Tests for Omega_h-backed Field operations."""

    @staticmethod
    def _build_mesh(world, dim, nx=10):
        """Build a box mesh of the requested dimension."""
        ny = nx if dim > 1 else 0
        nz = nx if dim > 2 else 0
        return omega_h.build_box(
            world, omega_h.Family.SIMPLEX, 1.0, 1.0, 1.0, nx, ny, nz, False,
        )

    @pytest.mark.parametrize("dim, order, num_components", [
        (2, 1, 1), (2, 2, 1),
    ])
    def test_field_methods(self, world, dim, order, num_components):
        """Create an Omega_h-backed Field and exercise the public Field API."""
        mesh = self._build_mesh(world, dim)
        factory = pcms.LagrangeFunctionSpace.from_mesh(
            mesh, order, num_components, pcms.CoordinateSystem.Cartesian
        )
        field = factory.create_field()

        assert field.get_num_components() == num_components
        assert field.get_num_dof_holders() > 0

        coords = field.get_dof_holder_coordinates()
        assert coords.shape[0] == field.get_num_dof_holders()
        assert coords.shape[1] == dim

        ndata = field.get_num_dof_holders() * num_components
        test_data = np.arange(ndata, dtype=np.float64)
        field.set_dof_holder_data(test_data)
        np.testing.assert_allclose(field.get_dof_holder_data(), test_data)

    @pytest.mark.parametrize("dim, order, num_components", [
        (2, 1, 1), (2, 2, 1),
    ])
    def test_field_transfer(self, world, dim, order, num_components):
        """Transfer data between identical Omega_h-backed function spaces."""
        if num_components != 1:
            pytest.skip("Multi-component transfer not yet supported")

        mesh = self._build_mesh(world, dim)
        source_space = pcms.LagrangeFunctionSpace.from_mesh(
            mesh, order, num_components, pcms.CoordinateSystem.Cartesian
        )
        target_space = pcms.LagrangeFunctionSpace.from_mesh(
            mesh, order, num_components, pcms.CoordinateSystem.Cartesian
        )
        source = source_space.create_field()
        target = target_space.create_field()

        coords = source.get_dof_holder_coordinates()
        source_data = np.zeros(source.get_num_dof_holders(), dtype=np.float64)
        for i in range(source.get_num_dof_holders()):
            source_data[i] = np.sum(coords[i, :dim])
        source.set_dof_holder_data(source_data)

        interp = pcms.Interpolator(source_space, target_space)
        interp.apply(source, target)
        np.testing.assert_allclose(
            target.get_dof_holder_data(), source_data, atol=1e-14
        )

    @pytest.mark.parametrize("dim, order, num_components", [
        (2, 1, 1), (2, 2, 1),
    ])
    def test_field_evaluation(self, world, dim, order, num_components):
        """Evaluate an Omega_h-backed field at explicit query points."""
        mesh = self._build_mesh(world, dim)
        factory = pcms.LagrangeFunctionSpace.from_mesh(
            mesh, order, num_components, pcms.CoordinateSystem.Cartesian
        )
        field = factory.create_field()

        mesh_coords = mesh.coords()
        num_verts = mesh.nverts()
        num_owned = field.get_num_dof_holders()
        ndata = num_owned * num_components

        def test_func(x, y, z, component):
            return np.sin(5.0 * x * y) + float(component)

        test_data = np.zeros(ndata, dtype=np.float64)
        for i in range(min(num_verts, num_owned)):
            x = mesh_coords[i * dim + 0] if dim >= 1 else 0.0
            y = mesh_coords[i * dim + 1] if dim >= 2 else 0.0
            z = mesh_coords[i * dim + 2] if dim >= 3 else 0.0
            for c in range(num_components):
                idx = i * num_components + c
                test_data[idx] = test_func(x, y, z, c)

        if order == 2 and num_owned > num_verts:
            for i in range(num_verts, num_owned):
                for c in range(num_components):
                    idx = i * num_components + c
                    test_data[idx] = float(c) + 0.5

        field.set_dof_holder_data(test_data)

        if dim == 2:
            eval_coords = np.array([
                [0.5, 0.5],
                [0.25, 0.25],
                [0.75, 0.75],
                [0.1, 0.9],
                [0.9, 0.1],
            ], dtype=np.float64)
        else:
            eval_coords = np.array([
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [0.1, 0.9, 0.1],
                [0.9, 0.1, 0.9],
            ], dtype=np.float64)

        request = pcms.EvaluationRequest.from_coordinates(eval_coords)
        evaluator = factory.create_point_evaluator(request)
        eval_values = np.zeros((eval_coords.shape[0], num_components),
                               dtype=np.float64)
        evaluator.evaluate(field, eval_values)

        for i in range(eval_coords.shape[0]):
            for c in range(num_components):
                val = eval_values[i, c]
                assert not np.isnan(val)
                assert not np.isinf(val)


class TestOmegaHTagOperations:
    """Tests for Omega_h mesh tag operations."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_tag_operations(self, world, dim):
        """Exercise Omega_h mesh tag creation, mutation, and query helpers."""
        nx, ny, nz = 5, (5 if dim > 1 else 0), (5 if dim > 2 else 0)
        mesh = omega_h.build_box(
            world, omega_h.Family.SIMPLEX, 1.0, 1.0, 1.0, nx, ny, nz, False,
        )
        rng = np.random.default_rng(42)
        nverts, nelems = mesh.nverts(), mesh.nelems()

        vertex_tag = rng.random(nverts).astype(np.float64)
        mesh.add_tag(0, "vertex_data", 1, vertex_tag)
        np.testing.assert_allclose(mesh.get_tag(0, "vertex_data"), vertex_tag)

        elem_quality = rng.random(nelems).astype(np.float64)
        mesh.add_tag(dim, "quality", 1, elem_quality)
        np.testing.assert_allclose(mesh.get_tag(dim, "quality"), elem_quality)

        if dim >= 2:
            edge_length = np.ones(mesh.nedges(), dtype=np.float64)
            mesh.add_tag(1, "edge_marker", 1, edge_length)
            np.testing.assert_allclose(
                mesh.get_tag(1, "edge_marker"), edge_length
            )

        assert len(mesh.ask_elem_verts()) > 0
        assert len(mesh.globals(0)) == nverts
        assert len(mesh.ask_verts_of(dim)) > 0
        assert len(mesh.owned(0)) == nverts
        assert np.sum(mesh.owned(0)) > 0
        assert isinstance(mesh.has_adj(0, dim), (bool, np.bool_))


class TestOmegaHEntityCoordinates:
    """Tests for entity-coordinate and averaging helpers on Omega_h meshes."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_entity_coordinates(self, world, dim):
        """Exercise entity-coordinate and averaging helpers."""
        nx, ny, nz = 4, (4 if dim > 1 else 0), (4 if dim > 2 else 0)
        mesh = omega_h.build_box(
            world, omega_h.Family.SIMPLEX, 1.0, 1.0, 1.0, nx, ny, nz, False,
        )
        vertex_coords = mesh.coords()
        assert len(vertex_coords) == mesh.nverts() * dim

        if dim >= 2:
            edge_coords = omega_h.average_field(mesh, 1, dim, vertex_coords)
            assert np.array(edge_coords).reshape(-1, dim).shape[0] == mesh.nedges()

        if dim == 2:
            elem_coords = omega_h.average_field(mesh, 2, dim, vertex_coords)
            assert np.array(elem_coords).reshape(-1, dim).shape[0] == mesh.nelems()

        if dim == 3:
            face_coords = omega_h.average_field(mesh, 2, dim, vertex_coords)
            assert np.array(face_coords).reshape(-1, dim).shape[0] == mesh.nfaces()
            region_coords = omega_h.average_field(mesh, 3, dim, vertex_coords)
            assert np.array(region_coords).reshape(-1, dim).shape[0] == mesh.nregions()

        vertex_field = np.arange(mesh.nverts(), dtype=np.float64)
        mesh.add_tag(0, "test_vertex_field", 1, vertex_field)
        averaged_field = omega_h.average_field(mesh, dim, 1, vertex_field)
        assert averaged_field.shape[0] == mesh.nents(dim)

