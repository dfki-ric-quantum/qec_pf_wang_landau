function(wl_enable_sanitizers target_name)
    option(ENABLE_COVERAGE "Enable coverage reporting")

    if(ENABLE_COVERAGE)
        target_compile_options(${target_name} INTERFACE --coverage -O0 -g)
        target_link_libraries(${target_name} INTERFACE --coverage)
    endif()

    set(SANITIZERS "")

    option(ENABLE_SANITIZER_ADDRESS "Address sanitizer" FALSE)
    if(ENABLE_SANITIZER_ADDRESS)
        list(APPEND SANITIZERS "address")
    endif()

    option(ENABLE_SANITIZER_UB "UB sanitizer" FALSE)
    if(ENABLE_SANITIZER_UB)
        list(APPEND SANITIZERS "undefined")
    endif()

    option(ENABLE_SANITIZER_THREAD "Thread sanitizer" FALSE)
    if(ENABLE_SANITIZER_THREAD)
        list(APPEND SANITIZERS "thread")
    endif()

    list(JOIN SANITIZERS "," LIST_OF_SANITIZERS)

    if(LIST_OF_SANITIZERS)
        if(NOT "${LIST_OF_SANITIZERS}" STREQUAL "")
            target_compile_options(${target_name} INTERFACE
                                   -fsanitize=${LIST_OF_SANITIZERS})
            target_link_libraries(${target_name} INTERFACE
                                   -fsanitize=${LIST_OF_SANITIZERS})
        endif()
    endif()
endfunction()
