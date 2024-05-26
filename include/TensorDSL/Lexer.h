#ifndef TENSORDSL_LEXER_H
#define TENSORDSL_LEXER_H

#include <string>
#include <vector>

namespace TensorDSL {

enum TokenType {
    TOKEN_EOF,
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_LBRACKET,
    TOKEN_RBRACKET,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_COMMA,
    TOKEN_ASSIGN,
    TOKEN_PLUS,
    TOKEN_SEMICOLON,
    TOKEN_PRINT
};

struct Token {
    TokenType type;
    std::string text;

    Token() : type(TOKEN_EOF), text("") {} // Default constructor
    Token(TokenType type, const std::string &text) : type(type), text(text) {}
};

class Lexer {
public:
    explicit Lexer(const std::string &source);
    Token getNextToken();

private:
    std::string source;
    size_t index;
    char currentChar;

    void advance();
    void skipWhitespace();
    Token identifier();
    Token number();
};

} // namespace TensorDSL

#endif // TENSORDSL_LEXER_H
