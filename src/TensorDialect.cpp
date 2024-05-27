#include "TensorDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tensor;

TensorDialect::TensorDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TensorDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "TensorOps.cpp.inc"
      >();
}
