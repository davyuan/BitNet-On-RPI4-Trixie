cmake_minimum_required(VERSION 3.10)
project(TestApp)

set(SAFETENSORS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../safetensors")
set(SAFETENSORS_LIB "${SAFETENSORS_DIR}/safetensors/target/release/libsafetensors.a")

# Verify the library file exists
if(NOT EXISTS "${SAFETENSORS_LIB}")
    message(FATAL_ERROR "SafeTensors library not found at: ${SAFETENSORS_LIB}")
endif()

message(STATUS "SafeTensors library: ${SAFETENSORS_LIB}")

add_executable(loadweight loadweight.cpp)

# Include the header directory
target_include_directories(loadweight PRIVATE ${SAFETENSORS_DIR}/include)

# Add define for safetensors C++ implementation
target_compile_definitions(loadweight PRIVATE SAFETENSORS_CPP_IMPLEMENTATION)

find_package(Threads REQUIRED)

# Link the static library using full path
target_link_libraries(loadweight PRIVATE 
    ${SAFETENSORS_LIB}
    Threads::Threads
    ${CMAKE_DL_LIBS}
)