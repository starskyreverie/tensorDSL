cmake_minimum_required(VERSION 3.10)
project(TensorDSL)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

include_directories(include)

add_executable(TensorDSL src/Main.cpp src/AST.cpp src/Lexer.cpp src/Parser.cpp src/TensorDialect.cpp)

llvm_update_compile_flags(TensorDSL)
target_link_libraries(TensorDSL MLIRTensorOps MLIRStandard MLIRSupport LLVMCore LLVMSupport)
