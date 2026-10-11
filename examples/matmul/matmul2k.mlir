// A single matmul layer, as a frontend (torch-mlir, ONNX importer, ...) would
// hand it to the compiler: linalg on tensors, results returned by value.
func.func @matmul(%lhs: tensor<1024x2048xbf16>, %rhs: tensor<2048x1024xbf16>) -> tensor<1024x1024xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1024x1024xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<1024x2048xbf16>, tensor<2048x1024xbf16>)
                          outs(%acc : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %result : tensor<1024x1024xf32>
}
