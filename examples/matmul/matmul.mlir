// A single matmul layer, as a frontend (torch-mlir, ONNX importer, ...) would
// hand it to the compiler: linalg on tensors, results returned by value.
func.func @matmul(%lhs: tensor<256x512xf32>, %rhs: tensor<512x256xf32>) -> tensor<256x256xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<256x256xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<256x512xf32>, tensor<512x256xf32>)
                          outs(%acc : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}
