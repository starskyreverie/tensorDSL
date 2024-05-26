#ifndef TENSORDSL_AST_H
#define TENSORDSL_AST_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <unordered_map>

namespace TensorDSL {

class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void evaluate(std::unordered_map<std::string, std::vector<std::vector<int>>>& tensorMap) const = 0;
};

class TensorDeclaration : public ASTNode {
public:
    std::string name;
    std::vector<std::vector<int>> values;
    
    TensorDeclaration(const std::string &name, const std::vector<std::vector<int>> &values)
        : name(name), values(values) {}

    void evaluate(std::unordered_map<std::string, std::vector<std::vector<int>>>& tensorMap) const override {
        tensorMap[name] = values;
        std::cout << "Declared Tensor " << name << " = [";
        for (const auto &row : values) {
            std::cout << "[";
            for (size_t i = 0; i < row.size(); ++i) {
                std::cout << row[i];
                if (i < row.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << "]" << std::endl;
    }
};

class TensorOperation : public ASTNode {
public:
    std::string result;
    std::string lhs;
    std::string rhs;
    char op;

    TensorOperation(const std::string &result, const std::string &lhs, const std::string &rhs, char op)
        : result(result), lhs(lhs), rhs(rhs), op(op) {}

    void evaluate(std::unordered_map<std::string, std::vector<std::vector<int>>>& tensorMap) const override {
        const auto& lhsTensor = tensorMap.at(lhs);
        const auto& rhsTensor = tensorMap.at(rhs);
        std::vector<std::vector<int>> resultTensor(lhsTensor.size(), std::vector<int>(lhsTensor[0].size(), 0));

        for (size_t i = 0; i < lhsTensor.size(); ++i) {
            for (size_t j = 0; j < lhsTensor[i].size(); ++j) {
                if (op == '+') {
                    resultTensor[i][j] = lhsTensor[i][j] + rhsTensor[i][j];
                }
                // Add more operations here if needed
            }
        }

        tensorMap[result] = resultTensor;
        std::cout << "Computed Tensor " << result << " = " << lhs << " " << op << " " << rhs << std::endl;
    }
};

class PrintOperation : public ASTNode {
public:
    std::string tensorName;

    PrintOperation(const std::string &tensorName) : tensorName(tensorName) {}

    void evaluate(std::unordered_map<std::string, std::vector<std::vector<int>>>& tensorMap) const override {
        auto it = tensorMap.find(tensorName);
        if (it != tensorMap.end()) {
            const auto& values = it->second;
            std::cout << "Tensor " << tensorName << " = [";
            for (const auto &row : values) {
                std::cout << "[";
                for (size_t i = 0; i < row.size(); ++i) {
                    std::cout << row[i];
                    if (i < row.size() - 1) std::cout << ", ";
                }
                std::cout << "]";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cerr << "Error: Tensor " << tensorName << " not found." << std::endl;
        }
    }
};

} // namespace TensorDSL

#endif // TENSORDSL_AST_H
