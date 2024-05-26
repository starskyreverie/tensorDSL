# TensorDSL

TensorDSL is a simple domain-specific language designed for tensor operations. This project includes a lexer, parser, AST, and basic compiler infrastructure using LLVM and MLIR.

**This project isn't anything serious.** It's mostly for fun and learning and isn't a full-fledged DSL.

## Directory Structure

- `include/TensorDSL/`: Header files for the lexer, parser, and AST.
- `src/`: Implementation files for the lexer, parser, and AST, along with the main entry point.

## Building the Project

1. **Install Dependencies**: Ensure LLVM and MLIR are installed on your system.
2. **Build the Project**:

    ```sh
    mkdir build
    cd build
    cmake ..
    make
    ```

3. **Run the Compiler**:

    ```sh
    ./TensorDSL <source_file>
    ```

## Overview

TensorDSL allows users to define tensor operations using a simple syntax. The code is parsed and converted into an Abstract Syntax Tree (AST), which is then translated to MLIR for optimization and finally lowered to LLVM IR for code generation.

### What is LLVM?

LLVM (Low-Level Virtual Machine) is a collection of modular and reusable compiler and toolchain technologies. It is designed for optimizing at compile-time, link-time, runtime, and "idle-time" and can generate machine code for various hardware platforms.

### What is MLIR?

MLIR (Multi-Level Intermediate Representation) is a flexible intermediate representation that can be used to represent different levels of abstraction in a compiler. It enables the optimization and transformation of high-level operations (like tensor computations) and lowers them to LLVM IR for efficient execution on hardware.

### How TensorDSL Uses LLVM and MLIR

1. **Parsing and AST Generation**: The TensorDSL source code is parsed into an Abstract Syntax Tree (AST).
2. **MLIR Dialect**: Custom MLIR dialects are defined to represent tensor operations.
3. **Optimization Passes**: MLIR optimization passes are applied to the tensor operations.
4. **Lowering to LLVM IR**: The optimized MLIR is lowered to LLVM IR.
5. **Code Generation**: LLVM is used to generate machine code from the LLVM IR, which is then executed.

## Example

### TensorDSL Code

```tensordsl
tensor A[2, 2] = [[1, 2], [3, 4]];
tensor B[2, 2] = [[5, 6], [7, 8]];
tensor C[2, 2] = A + B;
print(C);
```

You can run the example code by running the following command:

```sh
./TensorDSL example.tdsl
```

The output will be `[[6, 8], [10, 12]]`.

## License

This project is licensed under the MIT License.
