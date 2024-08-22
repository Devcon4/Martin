cmake_minimum_required(VERSION 3.10)

# Project name
project(Martin)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source files
add_executable(Martin src/main.cpp)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_subdirectory(external/vk-bootstrap)
add_subdirectory(external/spdlog)

include_directories(external/fmod/api/core/inc)

link_directories(external/fmod/api/core/lib/x86_64)

# Link libraries
target_link_libraries(Martin PRIVATE Vulkan::Vulkan vk-bootstrap spdlog::spdlog glfw)

function(compile_shader SHADER)
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(SPIRV "${CMAKE_BINARY_DIR}/resources/shaders/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/resources/shaders/"
        COMMAND glslc ${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${FILE_NAME}"
        VERBATIM
    )
    list(APPEND SPIRV_FILES ${SPIRV})
    set(SPIRV_FILES ${SPIRV_FILES} ${SPIRV} PARENT_SCOPE)
endfunction()

# Copy resources to output directory
file(COPY src/resources DESTINATION ${CMAKE_BINARY_DIR})

file(GLOB_RECURSE GLSL_SHADERS "src/resources/shaders/*.frag" "src/resources/shaders/*.vert")

foreach(GLSL_SHADER ${GLSL_SHADERS})
    compile_shader(${GLSL_SHADER})
endforeach()

add_custom_target(ShadersTarget DEPENDS ${SPIRV_FILES})

add_dependencies(Martin ShadersTarget)