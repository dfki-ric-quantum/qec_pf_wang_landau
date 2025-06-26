if(NOT DEFINED WL_GIT_INFO_SRC)
    set(WL_GIT_INFO_SRC ${PROJECT_SOURCE_DIR}/src/git_info/git_info.cpp.in)
endif()

if(NOT DEFINED WL_GIT_INFO_DST)
    set(WL_GIT_INFO_DST ${PROJECT_SOURCE_DIR}/src/git_info/git_info.cpp)
endif()

function(wl_config_git_info)
    find_package(Git)

    if(GIT_EXECUTABLE)
      execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE WL_GIT_VERSION
        RESULT_VARIABLE ERROR_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    endif()

    if(WL_GIT_VERSION STREQUAL "")
        set(WL_GIT_VERSION 0.0.0-unknown)
        message(WARNING "Failed to get git version, setting to \"${WL_GIT_VERSION}\".")
    endif()

    configure_file(
        ${WL_GIT_INFO_SRC}
        ${WL_GIT_INFO_DST}
        @ONLY)
endfunction()

if(WL_CONFIG_GIT_INFO)
    wl_config_git_info()
endif()
