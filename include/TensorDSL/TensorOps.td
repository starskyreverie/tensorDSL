include "mlir/IR/OpBase.td"

def Tensor_Dialect : Dialect {
  let name = "tensor";
}

def Tensor_AddOp : Tensor_Op<"add", []> {
  let summary = "addition of two tensors";
  let description = [{
    This operation adds two tensors element-wise.
  }];

  let arguments = (ins Tensor:$lhs, Tensor:$rhs);
  let results = (outs Tensor:$result);
}

def Tensor_PrintOp : Tensor_Op<"print", []> {
  let summary = "print a tensor";
  let description = [{
    This operation prints a tensor to the console.
  }];

  let arguments = (ins Tensor:$input);
  let results = (outs);
}
