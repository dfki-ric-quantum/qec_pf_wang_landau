add_custom_target(wl_git_info
    ${CMAKE_COMMAND} -D WL_CONFIG_GIT_INFO=1
                     -D WL_GIT_INFO_SRC=${PROJECT_SOURCE_DIR}/src/git_info/git_info.cpp.in
                     -D WL_GIT_INFO_DST=${PROJECT_SOURCE_DIR}/src/git_info/git_info.cpp
                     -P ${PROJECT_SOURCE_DIR}/cmake/git_info.cmake
)

add_library(git_info git_info.cpp)
target_link_libraries(git_info PUBLIC wl_options)
add_dependencies(git_info wl_git_info)
wl_config_git_info()
