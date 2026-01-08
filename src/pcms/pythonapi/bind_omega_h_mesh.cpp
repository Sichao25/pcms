#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Omega_h_mesh.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_comm.hpp>
#include "helpers.h"

namespace py = pybind11;

namespace pcms {

void bind_omega_h_mesh_module(py::module& m) {
  // Bind Omega_h::Library
  py::class_<Omega_h::Library, std::shared_ptr<Omega_h::Library>>(m, "OmegaHLibrary")
    .def(py::init<>(), "Default constructor")
    .def("world", &Omega_h::Library::world,
         py::return_value_policy::reference,
         "Get the world communicator");
  
  // Bind the Comm class
  py::class_<Omega_h::Comm, std::shared_ptr<Omega_h::Comm>>(m, "Comm")
      // Constructors
#ifdef OMEGA_H_USE_MPI
      .def(py::init<Omega_h::Library*, MPI_Comm>(),
            py::arg("library"),
            py::arg("impl"))
      .def(py::init([](Omega_h::Library* library, MPI_Comm impl, py::array_t<const Omega_h::I32> srcs,
                       py::array_t<const Omega_h::I32> dsts) {
        auto srcs_view = numpy_to_omega_h_read<Omega_h::I32>(srcs);
        auto dsts_view = numpy_to_omega_h_read<Omega_h::I32>(dsts);
        return new Omega_h::Comm(library, impl, srcs_view, dsts_view);
      }))
      .def("get_impl", &Omega_h::Comm::get_impl,
            "Get the underlying MPI communicator")
#else
      .def(py::init<Omega_h::Library*, bool, bool>(),
            py::arg("library"),
            py::arg("is_graph"),
            py::arg("sends_to_self"))
#endif
      // Methods
      .def("library", &Omega_h::Comm::library,
            py::return_value_policy::reference,
            "Get the library pointer")
      .def("rank", &Omega_h::Comm::rank,
            "Get the rank of this process")
      .def("size", &Omega_h::Comm::size,
            "Get the total number of processes")
      .def("dup", &Omega_h::Comm::dup,
            "Duplicate the communicator")
      .def("split", &Omega_h::Comm::split,
            py::arg("color"),
            py::arg("key"),
            "Split the communicator")
      .def("graph", &Omega_h::Comm::graph,
            py::arg("dsts"),
            "Create a graph communicator")
      .def("graph_adjacent", &Omega_h::Comm::graph_adjacent,
            py::arg("srcs"),
            py::arg("dsts"),
            "Create an adjacent graph communicator")
      .def("graph_inverse", &Omega_h::Comm::graph_inverse,
            "Get the inverse graph communicator")
      .def("sources", &Omega_h::Comm::sources,
            "Get source ranks")
      .def("destinations", &Omega_h::Comm::destinations,
            "Get destination ranks")
      .def("reduce_or", &Omega_h::Comm::reduce_or,
            py::arg("x"),
            "Reduce using logical OR")
      .def("reduce_and", &Omega_h::Comm::reduce_and,
            py::arg("x"),
            "Reduce using logical AND")
      .def("add_int128", &Omega_h::Comm::add_int128,
            py::arg("x"),
            "Add Int128 values across processes")
      .def("bcast_string", &Omega_h::Comm::bcast_string,
            py::arg("s"),
            py::arg("root_rank") = 0,
            "Broadcast a string")
      .def("barrier", &Omega_h::Comm::barrier,
            "Synchronize all processes");
  m.def("build_box", &Omega_h::build_box,
        py::arg("comm"),
        py::arg("family"),
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        py::arg("nx"),
        py::arg("ny"),
        py::arg("nz"),
        py::arg("symmetric") = false,
        "Build a box mesh with specified dimensions and discretization",
        py::return_value_policy::move);
  
  // Bind Omega_h::Mesh
  py::class_<Omega_h::Mesh, std::shared_ptr<Omega_h::Mesh>>(m, "OmegaHMesh")
    .def(py::init<>(), "Default constructor")
    .def(py::init<Omega_h::Library*>(), py::arg("library"),
         "Constructor with library")
    
    .def("set_library", &Omega_h::Mesh::set_library,
         py::arg("library"),
         "Set the library")
    
    .def("library", &Omega_h::Mesh::library,
         py::return_value_policy::reference,
         "Get the library")
    
    .def("set_comm", &Omega_h::Mesh::set_comm,
         py::arg("comm"),
         "Set the communicator")
    
    .def("comm", &Omega_h::Mesh::comm,
         "Get the communicator")
    
    .def("set_dim", &Omega_h::Mesh::set_dim,
         py::arg("dim"),
         "Set mesh dimension")
    
    .def("dim", &Omega_h::Mesh::dim,
         "Get mesh dimension")
    
    .def("set_family", &Omega_h::Mesh::set_family,
         py::arg("family"),
         "Set mesh family (simplex, hypercube, etc.)")
    
    .def("family", &Omega_h::Mesh::family,
         "Get mesh family")
    
    .def("nverts", &Omega_h::Mesh::nverts,
         "Get number of vertices")
    
    .def("nedges", &Omega_h::Mesh::nedges,
         "Get number of edges")
    
    .def("nfaces", &Omega_h::Mesh::nfaces,
         "Get number of faces")
    
    .def("nregions", &Omega_h::Mesh::nregions,
         "Get number of regions")
    
    .def("nelems", &Omega_h::Mesh::nelems,
         "Get number of elements")
    
    .def("nents", &Omega_h::Mesh::nents,
         py::arg("ent_dim"),
         "Get number of entities of given dimension")
    
    .def("nglobal_ents", &Omega_h::Mesh::nglobal_ents,
         py::arg("ent_dim"),
         "Get global number of entities")
    
    .def("coords", [](const Omega_h::Mesh& mesh) {
      auto coords = mesh.coords();
      // Convert to numpy array
      py::array_t<Omega_h::Real> result(coords.size());
      auto buf = result.request();
      Omega_h::Real* ptr = static_cast<Omega_h::Real*>(buf.ptr);
      for (Omega_h::LO i = 0; i < coords.size(); ++i) {
        ptr[i] = coords[i];
      }
      return result;
    },
    "Get mesh coordinates as numpy array")
    
    .def("set_coords", [](Omega_h::Mesh& mesh, py::array_t<Omega_h::Real> coords) {
      auto buf = coords.request();
      Omega_h::Write<Omega_h::Real> coords_write(buf.size);
      Omega_h::Real* ptr = static_cast<Omega_h::Real*>(buf.ptr);
      for (Omega_h::LO i = 0; i < buf.size; ++i) {
        coords_write[i] = ptr[i];
      }
      mesh.set_coords(Omega_h::Reals(coords_write));
    },
    py::arg("coords"),
    "Set mesh coordinates from numpy array")
    
    .def("add_coords", [](Omega_h::Mesh& mesh, py::array_t<Omega_h::Real> coords) {
      auto buf = coords.request();
      Omega_h::Write<Omega_h::Real> coords_write(buf.size);
      Omega_h::Real* ptr = static_cast<Omega_h::Real*>(buf.ptr);
      for (Omega_h::LO i = 0; i < buf.size; ++i) {
        coords_write[i] = ptr[i];
      }
      mesh.add_coords(Omega_h::Reals(coords_write));
    },
    py::arg("coords"),
    "Add mesh coordinates from numpy array")
    
    .def("has_tag", &Omega_h::Mesh::has_tag,
         py::arg("ent_dim"),
         py::arg("name"),
         "Check if mesh has a tag")
    
    .def("ntags", &Omega_h::Mesh::ntags,
         py::arg("ent_dim"),
         "Get number of tags for entity dimension")
    
    .def("remove_tag", &Omega_h::Mesh::remove_tag,
         py::arg("ent_dim"),
         py::arg("name"),
         "Remove a tag")
    
    .def("has_ents", &Omega_h::Mesh::has_ents,
         py::arg("ent_dim"),
         "Check if mesh has entities of given dimension")
    
    .def("ask_lengths", [](Omega_h::Mesh& mesh) {
      auto lengths = mesh.ask_lengths();
      py::array_t<Omega_h::Real> result(lengths.size());
      auto buf = result.request();
      Omega_h::Real* ptr = static_cast<Omega_h::Real*>(buf.ptr);
      for (Omega_h::LO i = 0; i < lengths.size(); ++i) {
        ptr[i] = lengths[i];
      }
      return result;
    },
    "Get edge lengths")
    
    .def("ask_qualities", [](Omega_h::Mesh& mesh) {
      auto qualities = mesh.ask_qualities();
      py::array_t<Omega_h::Real> result(qualities.size());
      auto buf = result.request();
      Omega_h::Real* ptr = static_cast<Omega_h::Real*>(buf.ptr);
      for (Omega_h::LO i = 0; i < qualities.size(); ++i) {
        ptr[i] = qualities[i];
      }
      return result;
    },
    "Get element qualities")
    
    .def("min_quality", &Omega_h::Mesh::min_quality,
         "Get minimum element quality")
    
    .def("max_length", &Omega_h::Mesh::max_length,
         "Get maximum edge length")
    
    .def("balance", py::overload_cast<bool>(&Omega_h::Mesh::balance),
         py::arg("predictive") = false,
         "Balance the mesh across processors")
    
    .def("set_parting", 
         py::overload_cast<Omega_h_Parting, bool>(&Omega_h::Mesh::set_parting),
         py::arg("parting"),
         py::arg("verbose") = false,
         "Set mesh partitioning")
    
    .def("parting", &Omega_h::Mesh::parting,
         "Get mesh partitioning type")
    
    .def("nghost_layers", &Omega_h::Mesh::nghost_layers,
         "Get number of ghost layers")
    
    .def("owned", [](Omega_h::Mesh& mesh, Omega_h::Int ent_dim) {
      auto owned = mesh.owned(ent_dim);
      py::array_t<Omega_h::I8> result(owned.size());
      auto buf = result.request();
      Omega_h::I8* ptr = static_cast<Omega_h::I8*>(buf.ptr);
      for (Omega_h::LO i = 0; i < owned.size(); ++i) {
        ptr[i] = owned[i];
      }
      return result;
    },
    py::arg("ent_dim"),
    "Get ownership flags for entities");
  
  // Bind mesh I/O functions
  m.def("read_mesh", 
        [](const std::string& filepath, Omega_h::Library* lib) {
          Omega_h::Mesh mesh(lib);
          Omega_h::binary::read(filepath, lib->world(), &mesh);
          return mesh;
        },
        py::arg("filepath"),
        py::arg("library"),
        "Read mesh from file");
  
  m.def("write_mesh",
        [](const std::string& filepath, Omega_h::Mesh& mesh) {
          Omega_h::binary::write(filepath, &mesh);
        },
        py::arg("filepath"),
        py::arg("mesh"),
        "Write mesh to file");

  // Other utilities
  py::enum_<Omega_h_Family>(m, "Family")
        .value("SIMPLEX", OMEGA_H_SIMPLEX)
        .value("HYPERCUBE", OMEGA_H_HYPERCUBE)
        .value("MIXED", OMEGA_H_MIXED)
        .export_values();  // This allows using values without the class prefix
  
  // Bind Omega_h_Parting enum
  py::enum_<Omega_h_Parting>(m, "OmegaHParting")
    .value("ELEM_BASED", OMEGA_H_ELEM_BASED)
    .value("VERT_BASED", OMEGA_H_VERT_BASED)
    .value("GHOSTED", OMEGA_H_GHOSTED)
    .export_values();
}

} // namespace pcms
