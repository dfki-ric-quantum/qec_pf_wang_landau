include(CTest)

# Sanitizer tests
option(SAN_CU_TESTS "Run compute-sanitizer on cuda tests" FALSE)
if(SAN_CU_TESTS)
    find_program(COMP_SAN compute-sanitizer)
    set(sanPassRegex "ERROR SUMMARY: 0 errors")
endif()

# Find python for integration tests
find_package(Python3 COMPONENTS Interpreter)

# Create compute sanitizer test for exiting CUDA unit test
function(wl_compute_sanitizer test_name)
    add_test(
        NAME sanitize_${test_name}
        COMMAND ${COMP_SAN} test_${test_name}
    )
    set_property(TEST sanitize_${test_name}
                 PROPERTY PASS_REGULAR_EXPRESSION "${sanPassRegex}")
    set_tests_properties(sanitize_${test_name} PROPERTIES LABELS sanitize)
endfunction()

# Create cuda unit test
function(wl_test_cu test_name)
    add_executable(test_${test_name} test_${test_name}.cu)
    configure_cuda(test_${test_name})
    target_link_libraries(test_${test_name} PRIVATE wl_options ${Boost_LIBRARIES} wl curand utils)
    add_test(test_${test_name} test_${test_name})
    set_tests_properties(test_${test_name} PROPERTIES LABELS unit)

    if(SAN_CU_TESTS AND COMP_SAN)
        wl_compute_sanitizer(${test_name})
    endif()
endfunction()

# Create C++ unit test
function(wl_test_cpp test_name)
    add_executable(test_${test_name} test_${test_name}.cpp)
    target_link_libraries(test_${test_name} PRIVATE wl_options utils)
    add_test(test_${test_name} test_${test_name})
    set_tests_properties(test_${test_name} PROPERTIES LABELS unit)
endfunction()

# Create integration test from python script
function(wl_integration_py test_name)
    add_test(
        NAME integration_${test_name}
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/integration_${test_name}.py ${PROJECT_SOURCE_DIR}
        WORKING_DIRECTORY /tmp
    )
    set_tests_properties(integration_${test_name} PROPERTIES LABELS integration)
endfunction()

# CUDA Tests
wl_test_cu(calc_energy)
wl_test_cu(device_ptr)
wl_test_cu(init_interactions)
wl_test_cu(init_lattice)
wl_test_cu(init_lattice_offsets)
wl_test_cu(wang_landau_pre_run)
wl_test_cu(check_energy_ranges)
wl_test_cu(init_indices)
wl_test_cu(reset_d_cond)
wl_test_cu(init_offsets_histogram)
wl_test_cu(check_interactions)
wl_test_cu(redistribute_g_values)
wl_test_cu(calc_average_log_g)
wl_test_cu(replica_exchange)
wl_test_cu(check_histogram)
wl_test_cu(wang_landau)

# C++ Tests
wl_test_cpp(interval)
wl_test_cpp(file_io)
wl_test_cpp(result_handling)

# Integration tests
if(Python3_FOUND)
    wl_integration_py(prerun_check_energies)
    wl_integration_py(prerun_check_interactions)
    wl_integration_py(prerun_matching_hist_energies)
    wl_integration_py(mainrun)
else()
    message(AUTHOR_WARNING "Could not find python interpreter, integration tests not added.")
endif()
