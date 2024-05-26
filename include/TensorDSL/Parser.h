#ifndef TENSORDSL_PARSER_H
#define TENSORDSL_PARSER_H

#include "Lexer.h"
#include "AST.h"
#include <memory>
#include <vector>

namespace TensorDSL {

class Parser {
public:
    explicit Parser(Lexer &lexer);
    std::vector<std::unique_ptr<ASTNode>> parse();

private:
    Lexer &lexer;
    Token currentToken;

    void advance();
    std::unique_ptr<TensorDeclaration> parseTensorDeclaration();
    std::unique_ptr<TensorOperation> parseTensorOperation();
    std::unique_ptr<ASTNode> parsePrint();
};

} // namespace TensorDSL

#endif // TENSORDSL_PARSER_H
