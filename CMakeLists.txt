cmake_minimum_required(VERSION 3.10)
project(TensorDSL)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include/TensorDSL)

add_executable(TensorDSL src/Main.cpp src/AST.cpp src/Lexer.cpp src/Parser.cpp)
