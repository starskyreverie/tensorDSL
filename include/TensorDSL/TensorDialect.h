#ifndef TENSORDSL_TENSORDIALECT_H
#define TENSORDSL_TENSORDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace tensor {

class TensorDialect : public Dialect {
public:
  explicit TensorDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tensor"; }
};

#define GET_OP_CLASSES
#include "TensorOps.h.inc"

} // namespace tensor
} // namespace mlir

#endif // TENSORDSL_TENSORDIALECT_H
