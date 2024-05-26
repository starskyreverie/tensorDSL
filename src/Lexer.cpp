#include "Lexer.h"

namespace TensorDSL {

Lexer::Lexer(const std::string &source) : source(source), index(0) {
    currentChar = source[index];
}

void Lexer::advance() {
    index++;
    if (index < source.size()) {
        currentChar = source[index];
    } else {
        currentChar = '\0';
    }
}

void Lexer::skipWhitespace() {
    while (isspace(currentChar)) {
        advance();
    }
}

Token Lexer::identifier() {
    std::string result;
    while (isalnum(currentChar) || currentChar == '_') {
        result += currentChar;
        advance();
    }
    if (result == "print") {
        return Token(TOKEN_PRINT, result);
    }
    return Token(TOKEN_IDENTIFIER, result);
}

Token Lexer::number() {
    std::string result;
    while (isdigit(currentChar)) {
        result += currentChar;
        advance();
    }
    return Token(TOKEN_NUMBER, result);
}

Token Lexer::getNextToken() {
    while (currentChar != '\0') {
        if (isspace(currentChar)) {
            skipWhitespace();
            continue;
        }
        if (isalpha(currentChar) || currentChar == '_') {
            return identifier();
        }
        if (isdigit(currentChar)) {
            return number();
        }
        switch (currentChar) {
            case '[': advance(); return Token(TOKEN_LBRACKET, "[");
            case ']': advance(); return Token(TOKEN_RBRACKET, "]");
            case '{': advance(); return Token(TOKEN_LBRACE, "{");
            case '}': advance(); return Token(TOKEN_RBRACE, "}");
            case ',': advance(); return Token(TOKEN_COMMA, ",");
            case '=': advance(); return Token(TOKEN_ASSIGN, "=");
            case '+': advance(); return Token(TOKEN_PLUS, "+");
            case ';': advance(); return Token(TOKEN_SEMICOLON, ";");
            default: advance(); return Token(TOKEN_EOF, "");
        }
    }
    return Token(TOKEN_EOF, "");
}

} // namespace TensorDSL
