#include "Lexer.h"
#include "Parser.h"
#include "TensorDialect.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Support/ToolUtilities.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <source_file>" << std::endl;
        return 1;
    }

    std::ifstream sourceFile(argv[1]);
    if (!sourceFile.is_open()) {
        std::cerr << "Error: Could not open source file " << argv[1] << std::endl;
        return 1;
    }

    std::string source((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
    std::cout << "Source code:\n" << source << std::endl;

    TensorDSL::Lexer lexer(source);
    TensorDSL::Parser parser(lexer);

    auto nodes = parser.parse();
    std::unordered_map<std::string, std::vector<std::vector<int>>> tensorMap;

    std::cout << "Evaluating AST nodes...\n";
    for (const auto &node : nodes) {
        node->evaluate(tensorMap);
    }

mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    std::cout << "Generating MLIR...\n";
    for (const auto &node : nodes) {
        node->codegen(context, builder);
    }

    module.print(llvm::outs());

    return 0;
}
