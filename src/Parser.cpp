#include "Parser.h"
#include <iostream>

namespace TensorDSL {

Parser::Parser(Lexer &lexer) : lexer(lexer), currentToken() { // Initialize currentToken
    advance();
}

void Parser::advance() {
    currentToken = lexer.getNextToken();
    std::cout << "Token: " << currentToken.text << std::endl;
}

std::vector<std::unique_ptr<ASTNode>> Parser::parse() {
    std::vector<std::unique_ptr<ASTNode>> nodes;
    while (currentToken.type != TOKEN_EOF) {
        if (currentToken.type == TOKEN_IDENTIFIER) {
            std::string id = currentToken.text;
            advance();
            if (currentToken.type == TOKEN_LBRACKET) {
                nodes.push_back(parseTensorDeclaration());
            } else if (currentToken.type == TOKEN_ASSIGN) {
                nodes.push_back(parseTensorOperation());
            }
        } else if (currentToken.type == TOKEN_PRINT) {
            nodes.push_back(parsePrint());
        }
        advance();
    }
    return nodes;
}

std::unique_ptr<TensorDeclaration> Parser::parseTensorDeclaration() {
    std::string name = currentToken.text;
    advance(); // Skip the identifier
    advance(); // Skip the '['
    
    std::vector<int> dimensions;
    while (currentToken.type == TOKEN_NUMBER) {
        dimensions.push_back(std::stoi(currentToken.text));
        advance();
        if (currentToken.type == TOKEN_COMMA) {
            advance();
        } else if (currentToken.type == TOKEN_RBRACKET) {
            break;
        }
    }
    advance(); // Skip the ']'
    advance(); // Skip the '='
    advance(); // Skip the '['

    std::vector<std::vector<int>> values;
    while (currentToken.type == TOKEN_LBRACKET) {
        std::vector<int> row;
        advance(); // Skip the '['
        while (currentToken.type == TOKEN_NUMBER) {
            row.push_back(std::stoi(currentToken.text));
            advance();
            if (currentToken.type == TOKEN_COMMA) {
                advance();
            } else if (currentToken.type == TOKEN_RBRACKET) {
                break;
            }
        }
        values.push_back(row);
        advance(); // Skip the ']'
        if (currentToken.type == TOKEN_COMMA) {
            advance();
        } else if (currentToken.type == TOKEN_RBRACE) {
            break;
        }
    }
    advance(); // Skip the ']'

    std::cout << "Parsed TensorDeclaration: " << name << std::endl;
    return std::make_unique<TensorDeclaration>(name, values);
}

std::unique_ptr<TensorOperation> Parser::parseTensorOperation() {
    std::string result = currentToken.text;
    advance(); // Skip the identifier
    advance(); // Skip the '='
    std::string lhs = currentToken.text;
    advance(); // Skip the lhs identifier
    char op = currentToken.text[0];
    advance(); // Skip the operator
    std::string rhs = currentToken.text;
    advance(); // Skip the rhs identifier

    std::cout << "Parsed TensorOperation: " << result << " = " << lhs << " " << op << " " << rhs << std::endl;
    return std::make_unique<TensorOperation>(result, lhs, rhs, op);
}

std::unique_ptr<ASTNode> Parser::parsePrint() {
    advance(); // Skip the 'print' keyword
    std::string tensorName = currentToken.text;
    std::cout << "Parsed PrintOperation: " << tensorName << std::endl;
    advance(); // Move to the next token after the tensor name
    return std::make_unique<PrintOperation>(tensorName);
}

} // namespace TensorDSL
