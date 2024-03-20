#ifndef TVM_RELAY_ONNX_PRINTER
#define TVM_RELAY_ONNX_PRINTER
#include <tvm/relay/base.h>

#include <iostream>
namespace tvm {
namespace relay {

void OnnxPrint(Expr e, const std::string& path_to_onnx);

}
}  // namespace tvm
#endif  // TVM_RELAY_ONNX_PRINTER