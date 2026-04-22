/**
 * @file test_proxy_coupling_xgc_server_overlap.cpp
 * @brief Test demonstrating the CORRECT way to handle overlap domain in coupling
 * 
 * KEY DIFFERENCES from test_proxy_coupling_xgc_server.cpp (the OLD way):
 * 
 * OLD WAY (test_proxy_coupling_xgc_server.cpp):
 *   1. Calls markServerOverlapRegion() which modifies mesh's ownership internally
 *   2. Passes IsModelEntInOverlap to XGCFunctionSpace (field layer)
 *   3. Relies on side effect of markServerOverlapRegion modifying mesh state
 *   4. No explicit coupling-level overlap control
 * 
 * NEW WAY (this file):
 *   1. Does NOT call markServerOverlapRegion (avoids hacky mesh modification)
 *   2. Still passes IsModelEntInOverlap to XGCFunctionSpace (field ownership)
 *   3. Additionally creates OverlapMask at Application level (coupling filter)
 *   4. Clean separation: field ownership (layout) + coupling overlap (application)
 * 
 * IMPORTANT: The overlap function serves TWO purposes:
 *   - Field layer: Defines which DOFs are "owned" by this process
 *   - Coupling layer: OverlapMask provides additional filtering for communication
 * 
 * The OverlapMask is a coupling property that works ON TOP OF field ownership!
 */

#include <iostream>
#include <pcms.h>
#include <pcms/utility/types.h>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <data/simple.h>

#include "test_support.h"
#include "pcms/field/function_space/xgc.h"
#include "pcms/field/layout/xgc.h"
#include "pcms/coupler/serializer/xgc.h"
#include "pcms/coupler/coupler.hpp"
#include "pcms/coupler/overlap_mask.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/function_space/lagrange.h"

using pcms::ConstructRCFromOmegaHMesh;
using pcms::GO;
using pcms::make_array_view;
using pcms::ReverseClassificationVertex;

static constexpr bool done = true;
namespace ts = test_support;

void xgc_coupler_with_overlap(MPI_Comm comm, Omega_h::Mesh& mesh, 
                               std::string_view cpn_file)
{
  // Setup coupler and partition
  pcms::Coupler cpl("proxy_couple_server", comm, true,
                     redev::Partition{ts::setupServerPartition(mesh, cpn_file)});
  const auto partition = std::get<redev::ClassPtn>(cpl.GetPartition());
  
  // Setup reverse classification
  ReverseClassificationVertex rc;
  if (mesh.has_tag(0, "simNumbering")) {
    rc = ConstructRCFromOmegaHMesh(mesh, "simNumbering");
  } else {
    rc = ConstructRCFromOmegaHMesh<GO>(mesh, "global", pcms::IndexBase::Zero);
  }

  // KEY DIFFERENCE: We do NOT call markServerOverlapRegion here!
  // That function hackily modifies the mesh's ownership. 
  // Instead, we'll use the overlap function directly when creating OverlapMask.
  
  auto* application = cpl.AddApplication("proxy_couple");

  constexpr int nplanes = 2;
  std::array<std::vector<GO>, nplanes> data;
  std::vector<pcms::FieldHandle<GO>> fields;
  
  for (int i = 0; i < nplanes; ++i) {
    data[i].resize(mesh.nverts());
    std::stringstream ss;
    ss << "xgc_gids_plane_" << i;
    
    // NEW WAY: Create overlap mask from the overlap function directly
    // The OverlapMask will call this function for each DOF based on classification
    auto overlap_mask = std::make_unique<pcms::OverlapMask>(
      mesh.nverts(), 
      [](int dim, int id) -> int8_t { 
        return ts::IsModelEntInOverlap{}(dim, id); 
      });
    
    // Set overlap at APPLICATION level BEFORE adding layout
    // This is the proper separation of concerns
    application->SetLayoutOverlapMask(ss.str(), std::move(overlap_mask));
    
    // Create function space - pass the same overlap function as before
    // This defines field-level ownership. The OverlapMask at the Application
    // level provides an additional coupling-specific filter on top of this.
    auto function_space = pcms::XGCFunctionSpace(
      rc, ts::IsModelEntInOverlap{}, static_cast<pcms::LO>(mesh.nverts()));
    
    // Add layout - it will use the overlap mask we set above
    application->AddLayout(ss.str(), function_space.GetLayout());
    
    // Create and add field
    auto field = function_space.CreateField<pcms::GO>(
      std::make_unique<pcms::XGCFieldData<pcms::GO>>(
        function_space.GetXGCLayout(), pcms::FieldMetadata{},
        make_array_view(data[i])));
    
    std::unique_ptr<pcms::FieldSerializer<GO>> serializer =
      std::make_unique<pcms::XGCFieldSerializer<GO>>(comm);
    
    fields.push_back(application->AddField(
      ss.str(), std::move(field), std::move(serializer)));
  }

  // Communication loop
  do {
    application->ReceivePhase([&]() {
      std::for_each(fields.begin(), fields.end(),
                    [](const pcms::FieldHandle<GO>& f) { f.Receive(); });
    });
    application->SendPhase([&]() {
      std::for_each(fields.begin(), fields.end(),
                    [](const pcms::FieldHandle<GO>& f) { f.Send(); });
    });
    application->ReceivePhase([&]() {
      std::for_each(fields.begin(), fields.end(),
                    [](const pcms::FieldHandle<GO>& f) { f.Receive(); });
    });
    application->SendPhase([&]() {
      std::for_each(fields.begin(), fields.end(),
                    [](const pcms::FieldHandle<GO>& f) { f.Send(); });
    });
  } while (!done);

  // Verification: Check field data that was actually communicated
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    std::cout << "\n=== Field Communication Verification (NEW approach) ===" << std::endl;
    std::cout << "Total DOFs in field: " << data[0].size() << std::endl;
    
    // Count how many field values are non-zero (indicating data was received)
    int received_count = 0;
    int zero_count = 0;
    for (size_t i = 0; i < data[0].size(); ++i) {
      if (data[0][i] != 0) {
        received_count++;
      } else {
        zero_count++;
      }
    }
    
    std::cout << "Field values received (non-zero): " << received_count << std::endl;
    std::cout << "Field values not received (zero): " << zero_count << std::endl;
    
    // Show sample of received field data
    std::cout << "\nSample field data (plane 0, first 20 values):" << std::endl;
    for (int i = 0; i < std::min(20, static_cast<int>(data[0].size())); ++i) {
      std::cout << "  DOF[" << i << "] = " << data[0][i] << std::endl;
    }
    
    // Count expected overlap DOFs based on classification
    int overlap_count = 0;
    auto class_dims = mesh.get_array<Omega_h::I8>(0, "class_dim");
    auto class_ids = mesh.get_array<Omega_h::ClassId>(0, "class_id");
    auto class_dims_h = Omega_h::HostRead(class_dims);
    auto class_ids_h = Omega_h::HostRead(class_ids);
    
    for (int i = 0; i < mesh.nverts(); ++i) {
      if (ts::IsModelEntInOverlap{}(class_dims_h[i], class_ids_h[i])) {
        overlap_count++;
      }
    }
    
    std::cout << "\nExpected overlap DOFs: " << overlap_count << std::endl;
    std::cout << "Layout communicator count: " << application->GetLayoutCommunicatorCount() << std::endl;
    std::cout << "======================================================\n" << std::endl;
  }

  Omega_h::vtk::write_parallel("proxy_couple_overlap", &mesh, mesh.dim());
}

void omegah_coupler_with_overlap(MPI_Comm comm, Omega_h::Mesh& mesh,
                                  std::string_view cpn_file)
{
  // Setup coupler and partition
  pcms::Coupler cpl("proxy_couple_server", comm, true,
                     redev::Partition{ts::setupServerPartition(mesh, cpn_file)});
  const auto partition = std::get<redev::ClassPtn>(cpl.GetPartition());
  auto* application = cpl.AddApplication("proxy_couple");
  
  // Setup numbering
  std::string numbering;
  if (mesh.has_tag(0, "simNumbering")) {
    numbering = "simNumbering";
  } else {
    Omega_h::Write<GO> gids(mesh.nverts());
    auto globals = mesh.globals(0);
    Omega_h::parallel_for(
      mesh.nverts(), OMEGA_H_LAMBDA(int i) { gids[i] = globals[i] + 1; });
    mesh.add_tag<GO>(0, "simNumbering", 1, Omega_h::Read(gids));
    numbering = "simNumbering";
  }

  constexpr int nplanes = 2;
  std::vector<pcms::FieldHandle<GO>> fields;
  
  for (int i = 0; i < nplanes; ++i) {
    std::stringstream ss;
    ss << "xgc_gids_plane_" << i;
    
    // NEW WAY: Create overlap mask from function directly
    // The OverlapMask will evaluate this for each DOF
    auto overlap_mask = std::make_unique<pcms::OverlapMask>(
      mesh.nverts(), 
      [](int dim, int id) -> int8_t { 
        return ts::IsModelEntInOverlap{}(dim, id); 
      });
    
    // Set at APPLICATION level BEFORE adding layout
    application->SetLayoutOverlapMask(ss.str(), std::move(overlap_mask));
    
    // Create function space - no overlap function needed here
    auto factory = pcms::LagrangeFunctionSpace::FromMesh(
      mesh, 1, 1, pcms::CoordinateSystem::Cartesian, numbering,
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    
    // Add layout - it will use the overlap mask
    application->AddLayout(ss.str(), factory.GetLayout());
    
    // Create and add field
    auto field = factory.CreateField<GO>(
      std::make_unique<pcms::SimpleFieldData<GO>>(
        factory.GetLayout(), pcms::FieldMetadata{}));
    
    std::unique_ptr<pcms::FieldSerializer<GO>> serializer =
      std::make_unique<pcms::FieldSerializer<GO>>();
    
    fields.push_back(application->AddField(
      ss.str(), std::move(field), std::move(serializer)));
  }
  
  // Communication loop
  do {
    application->ReceivePhase([&]() {
      std::for_each(fields.begin(), fields.end(),
                    [](const pcms::FieldHandle<GO>& f) { f.Receive(); });
    });
    application->SendPhase([&]() {
      std::for_each(fields.begin(), fields.end(),
                    [](const pcms::FieldHandle<GO>& f) { f.Send(); });
    });
    application->ReceivePhase([&]() {
      std::for_each(fields.begin(), fields.end(),
                    [](const pcms::FieldHandle<GO>& f) { f.Receive(); });
    });
    application->SendPhase([&]() {
      std::for_each(fields.begin(), fields.end(),
                    [](const pcms::FieldHandle<GO>& f) { f.Send(); });
    });
  } while (!done);
  
  Omega_h::vtk::write_parallel("proxy_couple_overlap", &mesh, mesh.dim());
}

int main(int argc, char** argv)
{
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  const int rank = world->rank();
  int size = world->size();
  
  if (argc != 4) {
    if (!rank) {
      std::cerr << "Usage: " << argv[0]
                << " </path/to/omega_h/mesh> "
                   "</path/to/partitionFile.cpn> "
                   "<coupler type (0 xgc, 1 omega-h)>\n";
    }
    exit(EXIT_FAILURE);
  }

  const auto meshFile = argv[1];
  const auto classPartitionFile = argv[2];
  int coupler_type = std::stoi(argv[3]);

  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(meshFile, lib.world(), &mesh);
  MPI_Comm mpi_comm = lib.world()->get_impl();
  
  if (coupler_type == 0) {
    if (size != 1) {
      if (!rank) {
        std::cerr << "XGC Adapter only works on 1 rank (not a distributed mesh "
                     "datastructure)\n";
      }
      std::abort();
    }
    xgc_coupler_with_overlap(mpi_comm, mesh, classPartitionFile);
  } else if (coupler_type == 1) {
    omegah_coupler_with_overlap(mpi_comm, mesh, classPartitionFile);
  } else {
    std::cerr << "Invalid coupler type. Choose 0 for XGC, 1 for Omega-h\n";
    std::abort();
  }
  
  return 0;
}
