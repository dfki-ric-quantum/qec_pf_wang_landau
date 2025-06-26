find_package(Boost REQUIRED
             COMPONENTS program_options
                        unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(-DBOOST_TEST_DYN_LINK)

find_package(HDF5 REQUIRED COMPONENTS CXX HL)
