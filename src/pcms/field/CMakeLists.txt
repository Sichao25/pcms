set(PCMS_FIELD_HEADERS
        coordinate.h
        coordinate_systems.h
        coordinate_transform.h
        field.h
        coordinate_system.h
        field_layout.h
        nodal_field_factory.h
        adapter/point_cloud/point_cloud_layout.h
        adapter/point_cloud/point_cloud.h
)
set(PCMS_FIELD_SOURCES
        nodal_field_factory.cpp
        adapter/point_cloud/point_cloud_layout.cpp
        adapter/point_cloud/point_cloud.cpp
)

if (PCMS_ENABLE_OMEGA_H)
    list(APPEND PCMS_FIELD_SOURCES
            lagrange_field_factory.cpp
            adapter/omega_h/omega_h_lagrange_layout.cpp
            adapter/uniform_grid/uniform_grid_field_layout.cpp
            adapter/uniform_grid/uniform_grid_field.cpp
    )
    list(APPEND
            PCMS_FIELD_HEADERS
            lagrange_field_factory.h
    )
    list(APPEND PCMS_FIELD_HEADERS
            adapter/omega_h/omega_h_lagrange_layout.h
            adapter/omega_h/omega_h_lagrange_field.h
            adapter/uniform_grid/uniform_grid_field_layout.h
            adapter/uniform_grid/uniform_grid_field.h
            adapter/uniform_grid/uniform_grid_binary_field.h
    )
endif ()

if (PCMS_ENABLE_MESHFIELDS)
    list(APPEND PCMS_FIELD_SOURCES
            adapter/meshfields/mesh_fields_adapter_layout.cpp
    )
    list(APPEND PCMS_FIELD_HEADERS
            adapter/meshfields/mesh_fields_adapter_layout.h
            adapter/meshfields/mesh_fields_adapter2.h
    )
endif ()

add_library(pcms_field ${PCMS_FIELD_SOURCES})
set_target_properties(pcms_field PROPERTIES
        OUTPUT_NAME pcmsfield
        EXPORT_NAME field)
target_sources(pcms_field PUBLIC
        FILE_SET field
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
        FILES ${PCMS_FIELD_HEADERS})
add_library(pcms::field ALIAS pcms_field)
target_compile_features(pcms_field PUBLIC cxx_std_20)

target_link_libraries(pcms_field PUBLIC
        pcms::utility
        pcms::localization)

if(PCMS_ENABLE_OMEGA_H)
    target_link_libraries(pcms_field PUBLIC Omega_h::omega_h)
endif()

if (PCMS_ENABLE_MESHFIELDS)
    target_link_libraries(pcms_field PUBLIC meshfields::meshfields)
endif ()

target_include_directories(pcms_field INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(
        TARGETS pcms_field
        EXPORT pcms_field-targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pcms/field
        FILE_SET field DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pcms
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
        EXPORT pcms_field-targets
        NAMESPACE pcms::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pcms)
