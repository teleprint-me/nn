cmake_minimum_required(VERSION 3.14)
project(nn VERSION 0.0.1)

# Add the ggml submodule directory
add_subdirectory(ggml)

# Define the main executable
add_executable(nn model.cpp)

# Link the ggml library
target_link_libraries(nn PUBLIC ggml)

# Specify include directories for this specific target
target_include_directories(nn PUBLIC 
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/ggml/src
    ${PROJECT_SOURCE_DIR}
)

# Set the output directory for built binaries
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Add examples
# add_executable(tensors examples/tensors/tensors.cpp)
# target_link_libraries(tensors PUBLIC ggml)
# target_include_directories(tensors PUBLIC 
#     ${PROJECT_SOURCE_DIR}/ggml/include/ggml
# )

# Set the output directory for built examples
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/examples)

# Set the output directory for the ggml shared object
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
