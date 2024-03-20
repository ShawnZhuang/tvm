/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/ir/tensor_type.h>
#include <tvm/node/reflection.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>

#include "../../ir/attr_functor.h"
#include "onnx.pb.h"
namespace tvm {

using namespace onnx;
namespace onnx_helper {

// template <typename K, typename V>
// class MemoCache {
//  public:
//   MemoCache(/* args */) {}
//   ~MemoCache() {}

//   bool Contain(const K& k) { return cache_.count(k); };
//   V Get(const K& k) {
//     ICHECK(Contain(k));
//     return cache_[k];
//   }
//   void Set(const K& k, )

//       private : std::unordered_map<K, V> cache_;
// };

class OnnxHelper {
 public:
  OnnxHelper() {}

  TensorProto_DataType ParserDataType(const DataType& dtype) {
    if (data_type_cache_.count(dtype)) {
      return data_type_cache_[dtype];
    }
    auto dtype_name = runtime::DLDataType2String(dtype);
    std::transform(dtype_name.begin(), dtype_name.end(), dtype_name.begin(), ::toupper);
    // bool status = TensorProto_DataType_Parse(dtype_name, &dtype_id);
    TensorProto_DataType dtype_id = TensorProto_DataType_UNDEFINED;
    if (data_type_remapping_.count(dtype_name)) {
      dtype_id = data_type_remapping_.at(dtype_name);
      data_type_cache_[dtype] = dtype_id;
    }
    return dtype_id;
  }
  ~OnnxHelper() {}

 private:
  std::unordered_map<DataType, onnx::TensorProto_DataType> data_type_cache_;
  const std::unordered_map<std::string, onnx::TensorProto_DataType> data_type_remapping_ = {
      {"FLOAT", TensorProto_DataType_FLOAT},
      {"FLOAT32", TensorProto_DataType_FLOAT},
      {"UINT8", TensorProto_DataType_UINT8},
      {"INT8", TensorProto_DataType_INT8},
      {"UINT16", TensorProto_DataType_UINT16},
      {"INT16", TensorProto_DataType_INT16},
      {"INT32", TensorProto_DataType_INT32},
      {"INT64", TensorProto_DataType_INT64},
      {"STRING", TensorProto_DataType_STRING},
      {"BOOL", TensorProto_DataType_BOOL},
      {"FLOAT16", TensorProto_DataType_FLOAT16},
      {"DOUBLE", TensorProto_DataType_DOUBLE},
      {"FLOAT64", TensorProto_DataType_DOUBLE},
      {"UINT32", TensorProto_DataType_UINT32},
      {"UINT64", TensorProto_DataType_UINT64},
      {"COMPLEX64", TensorProto_DataType_COMPLEX64},
      {"COMPLEX128", TensorProto_DataType_COMPLEX128},
      {"BFLOAT16", TensorProto_DataType_BFLOAT16},
      {"FLOAT8E4M3FN", TensorProto_DataType_FLOAT8E4M3FN},
      {"FLOAT8E4M3FNUZ", TensorProto_DataType_FLOAT8E4M3FNUZ},
      {"FLOAT8E5M2", TensorProto_DataType_FLOAT8E5M2},
      {"FLOAT8E5M2FNUZ", TensorProto_DataType_FLOAT8E5M2FNUZ},
      {"UINT4", TensorProto_DataType_UINT4},
      {"INT4", TensorProto_DataType_INT4},
  };
};

static OnnxHelper helper = OnnxHelper();

TensorProto_DataType ParserDataType(const DataType& dtype) { return helper.ParserDataType(dtype); }

void ParserShape(TensorShapeProto* proto, const Array<PrimExpr>& shape) {
  for (size_t i = 0; i < shape.size(); i++) {
    auto dim_i = shape[i].as<IntImmNode>();
    if (dim_i != nullptr) {
      proto->add_dim()->set_dim_value(dim_i->value);
    } else {
      proto->add_dim()->set_dim_param("Any");
    }
  }
  return;
}

void ParserValueInfo(ValueInfoProto* proto, const std::string& name, const TensorType& type) {
  proto->set_name(name);
  auto tensor_type_proto = proto->mutable_type()->mutable_tensor_type();
  tensor_type_proto->set_elem_type(ParserDataType(type->dtype));
  ParserShape(tensor_type_proto->mutable_shape(), type->shape);
  return;
};

void ParserTensor(TensorProto* proto, const std::string& name, const relay::Constant& tensor) {
  proto->set_name(name);
  proto->set_data_type(ParserDataType(tensor->data.DataType()));
  for (auto dim_i : tensor->data.Shape()) {
    proto->add_dims(dim_i);
  }
  return;
}

}  // namespace onnx_helper
namespace relay {

class AttrsProtoPrinter : public AttrVisitor {
 public:
  AttrsProtoPrinter(onnx::NodeProto* node) : node_(node) {}
  ~AttrsProtoPrinter() {}

  class AttributeValueVisitor : public AttrFunctor<void(const ObjectRef&, onnx::AttributeProto*)> {
   public:
    AttributeValueVisitor() {}
    ~AttributeValueVisitor() {}

    void VisitAttrDefault_(const Object* op, onnx::AttributeProto* proto) final { return; };
    void VisitAttr_(const ArrayNode* op, onnx::AttributeProto* proto) final {
      for (size_t i = 0; i < op->size(); i++) {
        VisitAttr(op->at(i), proto);
      }
    };
    void VisitAttr_(const tir::IntImmNode* op, onnx::AttributeProto* proto) final {
      proto->add_ints(op->value);
    };
    void VisitAttr_(const tir::FloatImmNode* op, onnx::AttributeProto* proto) final {
      proto->add_ints(op->value);
    };
    void VisitAttr_(const tir::StringImmNode* op, onnx::AttributeProto* proto) final {
      proto->add_strings(op->value);
    };
  };

  void Visit(const char* key, double* value) {
    auto proto = node_->add_attribute();
    proto->set_name(key);
    proto->add_floats(*value);
    return;
  };
  void Visit(const char* key, int64_t* value) {
    auto proto = node_->add_attribute();
    proto->set_name(key);
    proto->add_ints(*value);
    return;
  };
  void Visit(const char* key, uint64_t* value) {
    auto proto = node_->add_attribute();
    proto->set_name(key);
    proto->add_ints(*value);
    return;
  };
  void Visit(const char* key, int* value) {
    auto proto = node_->add_attribute();
    proto->set_name(key);
    proto->add_ints(*value);
    return;
  };
  void Visit(const char* key, bool* value) {
    auto proto = node_->add_attribute();
    proto->set_name(key);
    proto->add_ints(*value);
    return;
  };
  void Visit(const char* key, std::string* value) {
    auto proto = node_->add_attribute();
    proto->set_name(key);
    proto->add_strings(*value);
    return;
  };
  void Visit(const char* key, void** value) {
    LOG_FATAL;
    return;
  };
  void Visit(const char* key, DataType* value) {
    auto proto = node_->add_attribute();
    proto->set_name(key);

    proto->add_type_protos()->mutable_tensor_type()->set_elem_type(
        onnx_helper::ParserDataType(*value));

    return;
  };
  void Visit(const char* key, runtime::NDArray* value) {
    LOG_FATAL;
    return;
  };
  void Visit(const char* key, runtime::ObjectRef* value) {
    auto proto = node_->add_attribute();
    proto->set_name(key);
    auto ref = *value;

    if (ref->IsInstance<tvm::StringObj>()) {
      auto value = Downcast<tvm::String>(ref);
      proto->add_strings(value);
      return;
    }
    AttributeValueVisitor().VisitAttr(ref, proto);

    LOG_INFO << key << (*value);
    return;
  };

 private:
  NodeProto* node_;
};

class ProtoBufferPrinter : public MixedModeVisitor {
 private:
  GraphProto* graph_;
  ModelProto* model_;

  int name_count_ = 0;
  std::unordered_map<const ExprNode*, std::string> cache_;

 public:
  ProtoBufferPrinter() {
    model_ = new onnx::ModelProto();
    graph_ = model_->mutable_graph();
  }
  ~ProtoBufferPrinter() {
    delete (model_);
    model_ = nullptr;
  }

  void VisitExpr_(const VarNode* op) {
    auto proto = graph_->add_input();
    onnx_helper::ParserValueInfo(proto, op->name_hint(), Downcast<TensorType>(op->checked_type()));
    Memo(op, proto->name());
  };
  void VisitExpr_(const GlobalVarNode* op) {
    auto proto = graph_->add_input();
    onnx_helper::ParserValueInfo(proto, op->name_hint, Downcast<TensorType>(op->checked_type()));
    Memo(op, op->name_hint);
  };
  void VisitExpr_(const ConstantNode* op) {
    auto cref = GetRef<Constant>(op);
    auto name = GenerateName(cref);
    auto proto = graph_->add_initializer();
    onnx_helper::ParserTensor(proto, name, cref);

    Memo(op, name);
  };
  void VisitExpr_(const CallNode* op) {
    auto proto = graph_->add_node();
    auto name = GenerateName(GetRef<Call>(op));
    LOG_INFO << name;
    proto->set_name(name);
    proto->set_op_type(Downcast<Op>(op->op)->name);
    for (size_t i = 0; i < op->args.size(); i++) {
      auto input = GetName(op->args[i]);
      proto->add_input(input);
    }
    if (op->attrs.defined()) {
      auto attr_proto_printer = AttrsProtoPrinter(proto);
      const_cast<BaseAttrsNode*>(op->attrs.get())->VisitAttrs(&attr_proto_printer);
    }
    // link output tensor
    auto output_type = Downcast<TensorType>(op->checked_type());
    auto output_name = name + "_0_";
    onnx_helper::ParserValueInfo(graph_->add_value_info(), output_name, output_type);
    proto->add_output(output_name);
    Memo(op, output_name);
  };
  void VisitExpr_(const TupleNode* op){

  };
  void VisitExpr_(const TupleGetItemNode* op){

  };
  void OnnxPrint(Expr e, const std::string& path_to_onnx) {
    graph_->Clear();
    VisitExpr(e);
    model_->set_ir_version(onnx::IR_VERSION);
    std::ofstream of(path_to_onnx, std::ios::binary);
    auto ss = model_->SerializeAsString();
    of.write(ss.c_str(), ss.size());

    return;
  }

  std::string GenerateName(const Expr& e) {
    auto call_node = e.as<CallNode>();
    std::stringstream ss;
    if (call_node != nullptr) {
      ss << Downcast<Op>(call_node->op)->name << "_" << name_count_;
    } else {
      ss << e->GetTypeKey() << "_" << name_count_;
    }
    name_count_ += 1;
    return ss.str();
  }

  std::string GetName(Expr e) {
    ICHECK(memo_.count(e.get()));
    return memo_[e.get()];
  };
  void Memo(const ExprNode* expr_node, const std::string& name) { memo_[expr_node] = name; }
  void Memo(const Expr& e, const std::string& name) { Memo(e.get(), name); }

  std::unordered_map<const ExprNode*, std::string> memo_;
};

void OnnxPrint(Expr e, const std::string& path_to_onnx) {
  LOG_INFO << PrettyPrint(e);
  auto gen = ProtoBufferPrinter();

  gen.OnnxPrint(e, path_to_onnx);
  return;
};

TVM_REGISTER_GLOBAL("relay.ir.OnnxPrint").set_body_typed(OnnxPrint);

}  // namespace relay

}  // namespace tvm