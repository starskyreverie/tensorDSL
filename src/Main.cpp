#include "Lexer.h"
#include "Parser.h"
#include <iostream>
#include <fstream>
#include <unordered_map>

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

    return 0;
}
