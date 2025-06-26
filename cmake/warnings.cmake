function(wl_set_warnings target_name)
    option(WARNINGS_AS_ERRORS "Treat warnings as errors" FALSE)

    set(CLANG_WARNINGS
        -Wall                  # All warning
        -Wextra                # Really all warnings
        -Wshadow               # warn if declaration gets shadowed by new declaration
        -Wnon-virtual-dtor     # warn about non virtual destructors
        -Wold-style-cast       # warn for c-style casts
        -Wcast-align           # warn for potential performance problem casts
        -Wunused               # warn on anything being unused
        -Woverloaded-virtual   # warn if you overload a virtual function
        -Wpedantic             # warn if non-standard C++ is used
        -Wconversion           # warn on type conversions that may lose data
        -Wsign-conversion      # warn on sign conversions
        -Wnull-dereference     # warn if a null dereference is detected
        -Wdouble-promotion     # warn if float is implicit promoted to double
        -Wformat=2             # warn on security issues around functions that format output
        -Wimplicit-fallthrough # warn on statements that fallthrough without an explicit annotation
    )


    set(GCC_WARNINGS
        ${CLANG_WARNINGS}
        -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
        -Wduplicated-cond        # warn if if / else chain has duplicated conditions
        -Wduplicated-branches    # warn if if / else branches have duplicated code
        -Wlogical-op             # warn about logical operations being used instead of bitwise
        -Wuseless-cast           # warn if you perform a cast to the same type
        -Wsuggest-override       # warn if an overridden member function is not marked
    )

    set(MSVC_WARNINGS
        /W4          # Standard warnings
        /w14242      # conversion with potential loss of data
        /w14254      # bitfield related conversion with potential loss of data
        /w14263      # member function does not override any base class virtual member function
        /w14265      # non-virtual destructor
        /w14287      # unsigned/negative constant mismatch
        /we4289      # loop control variable declared in the for-loop is used outside loop scope
        /w14296      # expression is always true/false
        /w14311      # pointer truncation
        /w14545      # expression before comma evaluates to a function which is missing an argument list
        /w14546      # function call before comma missing argument list
        /w14547      # operator before comma has no effect
        /w14549      # operator before comma has no effect
        /w14555      # expression has no effect
        /w14619      # pragma warning: there is no warning number 'number'
        /w14640      # Enable warning on thread un-safe static member initialization
        /w14826      # Conversion from 'type1' to 'type2' is sign-extended.
        /w14905      # wide string literal cast to 'LPSTR'
        /w14906      # string literal cast to 'LPWSTR'
        /w14928      # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
        /permissive- # standards conformance mode for MSVC compiler.
    )

    set(CUDA_WARNINGS
        -Wall
        -Wextra
        -Wunused
        -Wconversion
        -Wshadow
    )

    if(WARNINGS_AS_ERRORS)
        message(TRACE "Warnings are treated as errors")
        list(APPEND CLANG_WARNINGS -Werror)
        list(APPEND GCC_WARNINGS -Werror)
        list(APPEND MSVC_WARNINGS /WX)
    endif()

    if(MSVC)
        set(PROJECT_WARNINGS_CXX ${MSVC_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(PROJECT_WARNINGS_CXX ${CLANG_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(PROJECT_WARNINGS_CXX ${GCC_WARNINGS})
    else()
        message(AUTHOR_WARNING "No compiler warnings set for CXX compiler: '${CMAKE_CXX_COMPILER_ID}'")
    endif()

    set(PROJECT_WARNINGS_C "${PROJECT_WARNINGS_CXX}")
    set(PROJECT_WARNINGS_CUDA "${CUDA_WARNINGS}")

    target_compile_options(${target_name}
        INTERFACE # C++ warnings
                  $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS_CXX}>
                  # C warnings
                  $<$<COMPILE_LANGUAGE:C>:${PROJECT_WARNINGS_C}>
                  # Cuda warnings
                  $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_WARNINGS_CUDA}>)
endfunction()
