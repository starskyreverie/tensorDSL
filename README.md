# TensorDSL

TensorDSL is a domain-specific language designed for performing tensor operations. This project includes a lexer, parser, Abstract Syntax Tree (AST), and a basic compiler infrastructure utilizing both LLVM and MLIR. The main objective is to explore the intricacies of compiler construction and learn about LLVM and MLIR through a practical implementation.

## Directory Structure

- `include/TensorDSL/`: Contains the header files for the lexer, parser, AST, and MLIR dialects.
  - `AST.h`: Defines the AST nodes and their functionalities.
  - `Lexer.h`: Declares the lexer for tokenizing the input source code.
  - `Parser.h`: Declares the parser for generating the AST from tokens.
  - `TensorDialect.h`: Declares the custom MLIR dialect and operations.
  - `TensorOps.h.inc`: Generated header file for MLIR operations.
  - `TensorOps.td`: TableGen description of the custom MLIR operations.
- `src/`: Contains the implementation files for the lexer, parser, AST, MLIR dialects, and the main entry point.
  - `AST.cpp`: Implements the functionalities of AST nodes.
  - `Lexer.cpp`: Implements the lexer.
  - `Main.cpp`: The main entry point that drives the compilation process.
  - `Parser.cpp`: Implements the parser.
  - `TensorDialect.cpp`: Implements the custom MLIR dialect and operations.

## Building the Project

### Prerequisites

Ensure LLVM and MLIR are installed on your system. If not, follow these steps:

1. **Clone the LLVM Project**:

    ```sh
    git clone https://github.com/llvm/llvm-project.git
    cd llvm-project
    ```

2. **Create a Build Directory**:

    ```sh
    mkdir build
    cd build
    ```

3. **Configure the Build with CMake**:

    ```sh
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DCMAKE_INSTALL_PREFIX=../install
    ```

4. **Build and Install**:

    ```sh
    ninja
    ninja install
    ```

5. **Set Up Environment Variables**:

    ```sh
    export PATH=/path/to/llvm-project/install/bin:$PATH
    export CMAKE_PREFIX_PATH=/path/to/llvm-project/install:$CMAKE_PREFIX_PATH
    ```

### Building TensorDSL

1. **Clone the TensorDSL Repository**:

    ```sh
    git clone <repository_url>
    cd TensorDSL
    ```

2. **Create a Build Directory**:

    ```sh
    mkdir build
    cd build
    ```

3. **Configure the Project with CMake**:

    ```sh
    cmake ..
    ```

4. **Build the Project**:

    ```sh
    make
    ```

## Running the Compiler

Run the TensorDSL compiler with a source file:

```sh
./TensorDSL <source_file>
```

For example, to run the provided example:

```sh
./TensorDSL example.tdsl
```

The output will be `[[6, 8], [10, 12]]`.

## Overview

### Parsing and AST Generation

The source code written in TensorDSL is first tokenized by the lexer. The lexer converts the raw input into a series of tokens, each representing a syntactic element such as identifiers, numbers, or operators. This process is essential for breaking down the high-level source code into manageable parts.

The parser then processes these tokens to generate an Abstract Syntax Tree (AST). The AST represents the hierarchical structure of the source code, capturing the semantic meaning of the program. Each node in the AST corresponds to a construct in the TensorDSL language, such as tensor declarations, operations, or print statements.

#### Lexer (include/TensorDSL/Lexer.h, src/Lexer.cpp)

The lexer reads the input source code character by character and groups them into tokens. Each token is classified into types such as `TOKEN_IDENTIFIER`, `TOKEN_NUMBER`, `TOKEN_ASSIGN`, etc. This classification helps the parser in understanding the structure and semantics of the code.

- **Methods**:
  - `getNextToken()`: Advances the lexer to the next token in the input.
  - `isIdentifierChar(char)`: Determines if a character is valid in an identifier.
  - `isNumberChar(char)`: Determines if a character is part of a number.

#### Parser (include/TensorDSL/Parser.h, src/Parser.cpp)

The parser takes the tokens produced by the lexer and builds the AST. It uses recursive descent parsing, a top-down parsing technique where each grammar rule is a function. The parser ensures that the tokens follow the correct syntax defined by TensorDSL.

- **Methods**:
  - `parse()`: Entry point for parsing the entire source code.
  - `parseTensorDeclaration()`: Parses tensor declaration statements.
  - `parseTensorOperation()`: Parses tensor operation statements.
  - `parsePrint()`: Parses print statements.

### MLIR Dialect

TensorDSL defines a custom MLIR dialect to represent tensor operations. This dialect includes operations such as tensor addition and tensor printing. The dialect provides a high-level abstraction suitable for various optimizations and transformations within the MLIR framework.

#### TensorDialect (include/TensorDSL/TensorDialect.h, src/TensorDialect.cpp)

Defines the custom MLIR dialect for tensor operations. This dialect includes operations like `tensor.add` for addition and `tensor.print` for printing tensor values.

- **Key Components**:
  - **TensorDialect Class**: Registers the operations and types associated with the tensor dialect.
  - **Operations**: Defined using TableGen in `TensorOps.td` and generated into `TensorOps.h.inc` and `TensorOps.cpp.inc`.

#### TensorOps (include/TensorDSL/TensorOps.td)

TableGen description of the custom MLIR operations. This file defines the syntax and semantics of the tensor operations in the dialect.

### Optimization Passes

Once the AST is generated, it is translated into MLIR. MLIR provides a flexible intermediate representation that supports multiple levels of abstraction. Optimization passes can be applied to the MLIR representation to improve performance and efficiency.

#### MLIR Code Generation (AST.h, AST.cpp)

The AST nodes are responsible for generating MLIR code. Each node type has a `codegen` method that produces the corresponding MLIR operations.

- **Methods**:
  - `TensorDeclaration::codegen()`: Generates MLIR code for tensor declarations.
  - `TensorOperation::codegen()`: Generates MLIR code for tensor operations.
  - `PrintOperation::codegen()`: Generates MLIR code for print operations.

### Lowering to LLVM IR

After optimization, the MLIR is lowered to LLVM IR. This process translates the high-level operations into lower-level operations that can be executed efficiently on the hardware. LLVM IR is a low-level intermediate representation that is closer to machine code.

#### MLIR to LLVM Conversion

MLIR provides conversion passes that transform MLIR operations into their equivalent LLVM IR operations. This involves mapping high-level tensor operations to lower-level LLVM operations.

- **Passes**:
  - **LowerToLLVM**: Converts custom MLIR operations to LLVM IR.

### Code Generation

Finally, LLVM is used to generate the machine code from the LLVM IR. This machine code can then be executed to perform the tensor operations specified in the original TensorDSL source code.

#### LLVM Code Generation

LLVM optimizes and generates machine code for the target architecture. The machine code generation involves several stages, including instruction selection, register allocation, and assembly code generation.

- **Components**:
  - **LLVMContext**: Manages LLVM IR objects.
  - **IRBuilder**: Facilitates the construction of LLVM IR instructions.
  - **Module**: Represents the entire program in LLVM IR.

## Detailed Explanation of Each File

### include/TensorDSL/AST.h

Defines the AST nodes and their functionalities. Each node type (e.g., tensor declaration, tensor operation, print operation) is represented by a class that inherits from the base `ASTNode` class. These classes provide methods for evaluating the node (i.e., performing the corresponding computation) and for generating MLIR code.

- **Classes**:
  - `ASTNode`: Base class for all AST nodes.
  - `TensorDeclaration`: Represents a tensor declaration.
  - `TensorOperation`: Represents a tensor operation (e.g., addition).
  - `PrintOperation`: Represents a print statement.

### include/TensorDSL/Lexer.h

Declares the lexer, which is responsible for tokenizing the input source code. The lexer processes the input string character by character, identifying syntactic elements such as keywords, identifiers, numbers, and operators, and producing a corresponding sequence of tokens.

- **Classes**:
  - `Token`: Represents a single token with type and text.
  - `Lexer`: Tokenizes the input source code.

### include/TensorDSL/Parser.h

Declares the parser, which generates the AST from the sequence of tokens produced by the lexer. The parser uses recursive descent parsing to process the tokens and construct the hierarchical AST structure.

- **Classes**:
  - `Parser`: Parses tokens and constructs the AST.

### include/TensorDSL/TensorDialect.h

Declares the custom MLIR dialect and operations. The dialect defines the set of operations and types specific to TensorDSL, providing a high-level abstraction for tensor computations.

- **Classes**:
  - `TensorDialect`: Registers the custom tensor operations and types.

### include/TensorDSL/TensorOps.h.inc

Generated header file for MLIR operations. This file is generated using MLIR TableGen and contains declarations for the operations defined in `TensorOps.td`.

### include/TensorDSL/TensorOps.td

TableGen description of the custom MLIR operations. This file defines the operations in the Tensor dialect using the TableGen language, which is a domain-specific language for defining compiler constructs.

### src/AST.cpp

Implements the functionalities of AST nodes. Each node type provides an `evaluate` method for performing the corresponding computation and a `codegen` method for generating MLIR code.

- **Methods**:
  - `TensorDeclaration::evaluate()`: Evaluates a tensor declaration.
  - `TensorOperation::evaluate()`: Evaluates a tensor operation.
  - `PrintOperation::evaluate()`: Evaluates a print operation.

### src/Lexer.cpp

Implements the lexer. The lexer processes the input string, identifies syntactic elements, and produces a sequence of tokens. Each token is represented by an instance of the `Token` class.

- **Methods**:
  - `Lexer::getNextToken()`: Advances the lexer to the next token.
  - `Lexer::isIdentifierChar(char)`: Determines if a character is valid in an identifier.
  - `Lexer::isNumberChar(char)`: Determines if a character is part of a number.

### src/Main.cpp

The main entry point for the TensorDSL compiler. This file drives the compilation process by initializing the lexer and parser, generating the AST, and invoking the MLIR code generation and LLVM code generation steps.

- **Methods**:
  - `main()`: Initializes components and runs the compilation pipeline.

### src/Parser.cpp

Implements the parser. The parser processes the sequence of tokens produced by the lexer and constructs the AST. The parser uses recursive descent parsing to handle the various syntactic constructs of TensorDSL.

- **Methods**:
  - `Parser::parse()`: Parses the entire source code.
  - `Parser::parseTensorDeclaration()`: Parses tensor declaration statements.
  - `Parser::parseTensorOperation()`: Parses tensor operation statements.
  - `Parser::parsePrint()`: Parses print statements.

### src/TensorDialect.cpp

Implements the custom MLIR dialect and operations. This file defines the `TensorDialect` class and the operations in the Tensor dialect. The dialect provides a high-level abstraction for tensor computations and serves as the target for MLIR code generation.

- **Methods**:
  - `TensorDialect::TensorDialect()`: Registers the operations and types in the tensor dialect.

## Example

### TensorDSL Code

The following example defines two tensors, `A` and `B`, each with dimensions 2x2. The tensors are initialized with specific values. A new tensor, `C`, is defined as the element-wise sum of `A` and `B`. The value of tensor `C` is then printed.

```tensordsl
tensor A[2, 2] = [[1, 2], [3, 4]];
tensor B[2, 2] = [[5, 6], [7, 8]];
tensor C[2, 2] = A + B;
print(C);
