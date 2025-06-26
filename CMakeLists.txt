cmake_minimum_required(VERSION 3.20)
project(wl-decoder VERSION 0.1
                   DESCRIPTION "Surface code decoder based on the Wang-Landau algorithm"
                   LANGUAGES CXX CUDA)

include(cmake/settings.cmake)
include(cmake/sanitizers.cmake)
include(cmake/static_analyzers.cmake)
include(cmake/warnings.cmake)
include(cmake/cuda.cmake)
include(cmake/dependencies.cmake)
include(cmake/git_info.cmake)


option(ENABLE_WARNINGS "Turn on all compiler warnings" TRUE)

# Warnings
add_library(wl_options INTERFACE)

if(ENABLE_WARNINGS)
    wl_set_warnings(wl_options)
else()
    message(WARNING "Warnings are turned off")
endif()
wl_enable_sanitizers(wl_options)

enable_testing()

add_subdirectory(src)
add_subdirectory(test)

install(TARGETS prerun mainrun)
