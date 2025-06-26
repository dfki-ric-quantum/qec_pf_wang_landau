add_subdirectory(git_info/)

add_library(wl wl.cu)
configure_cuda(wl)
target_link_libraries(wl PUBLIC wl_options curand)

add_library(utils interval.cpp
                  program_options.cpp
                  result_handling.cpp
                  utils.cpp
                  file_io.cpp)
target_link_libraries(utils PUBLIC wl_options ${Boost_LIBRARIES} HDF5::HDF5 git_info)

# Prerun
add_executable(prerun prerun.cu)
configure_cuda(prerun)
target_link_libraries(prerun PRIVATE wl_options wl utils)

# Main run
add_executable(mainrun mainrun.cu)
configure_cuda(mainrun)
target_link_libraries(mainrun PRIVATE wl_options wl utils)
