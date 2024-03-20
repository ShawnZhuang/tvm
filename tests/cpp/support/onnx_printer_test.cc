
#include <gtest/gtest.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>
#include <tvm/relay/transform.h>
#include <tvm/support/onnx_printer.h>
namespace tvm {
namespace relay {

TEST(ProtoBuf, SimpleTest) {
  constexpr const char* kModel = R"(
    #[version = "0.0.5"]
    def @main(%data : Tensor[(1, 304, 128, 128), float32],
             %weight1 : Tensor[(304, 1, 3, 3), float32],
             %bias1 : Tensor[(304), float32],
             %weight2 : Tensor[(256, 304, 1, 1), float32],
             %bias2 : Tensor[(256), float32]) -> Tensor[(1, 256, 128, 128), float32] {
      %0 = nn.conv2d(%data, %weight1, padding=[1, 1, 1, 1], groups=304, channels=304, kernel_size=[3, 3]);
      %1 = nn.bias_add(%0, %bias1);
      %2 = nn.relu(%1);
      %3 = nn.conv2d(%2, %weight2, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
      %4 = nn.bias_add(%3, %bias2);
      nn.relu(%4)
    }
  )";

  IRModule module = ParseModule("string", kModel);
  auto opti = transform::Sequential({transform::InferType()});
  opti(module);
  auto func = module->Lookup("main");
  relay::OnnxPrint(func, "simple.onnx");
}

}  // namespace relay
}  // namespace tvm