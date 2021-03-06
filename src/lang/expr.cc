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

/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr.cc
 */
#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/expr_operator.h>
#include <ir/IRPrinter.h>
#include <memory>

namespace tvm {

using HalideIR::IR::RangeNode;

Range::Range(Expr begin, Expr end)
    : Range(make_node<RangeNode>(
          begin,
          is_zero(begin) ? end : (end - begin))) {
}

Range Range::make_by_min_extent(Expr min, Expr extent) {
  return Range(make_node<HalideIR::IR::RangeNode>(min, extent));
}

IterVar IterVarNode::make(Range dom, Var var,
                          IterVarType t, std::string thread_tag) {
  NodePtr<IterVarNode> n = make_node<IterVarNode>();
  n->dom = dom;
  n->var = var;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  return IterVar(n);
}

IterVar thread_axis(Range dom, std::string tag) {
  return IterVarNode::make(
      dom, Var(tag), kThreadIndex, tag);
}

IterVar reduce_axis(Range dom, std::string name) {
  return IterVarNode::make(
      dom, Var(name), kCommReduce);
}

std::ostream& operator<<(std::ostream& os, const NodeRef& n) {  // NOLINT(*)
  IRPrinter(os).print(n);
  return os;
}

void Dump(const NodeRef& n) {
  std::cerr << n << "\n";
}

Var var(const std::string& name_hint, Type t) {
  return Var(name_hint, t);
}

Voxel VoxelNode::make(  Array <IterVar> base,
              Array <IterVar> extern_before,
              Array <IterVar> extern_after)
{
    NodePtr<VoxelNode> n = make_node<VoxelNode>();
    n->base=base;
    n->extern_before=extern_before;
    n->extern_after=extern_after;
    return Voxel(n);
}




TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IterVarNode>([](const IterVarNode *op, IRPrinter *p) {
    p->stream << "iter_var(";
    if (op->var->name_hint.length() != 0) {
      p->stream  << op->var->name_hint << ", ";
    }
    if (op->dom.defined()) {
      p->stream << op->dom;
    }
    if (op->thread_tag.length() != 0) {
      p->stream << ", " << op->thread_tag;
    }
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<RangeNode>([](const HalideIR::IR::RangeNode *op, IRPrinter *p) {
    p->stream << "range(min=" << op->min << ", ext=" << op->extent << ')';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<VoxelNode>([](const VoxelNode *op, IRPrinter *p) {
  p->stream << "Voxel{base:" << op->base <<", before:" <<op->extern_before<<", after:"<<op->extern_after <<"}" ;
});

TVM_REGISTER_NODE_TYPE(ArrayNode);
TVM_REGISTER_NODE_TYPE(MapNode);
TVM_REGISTER_NODE_TYPE(StrMapNode);
TVM_REGISTER_NODE_TYPE(RangeNode);
TVM_REGISTER_NODE_TYPE(IterVarNode);
TVM_REGISTER_NODE_TYPE(VoxelNode);

}  // namespace tvm
