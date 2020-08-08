#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>

#include "../../arith/interval_set.h"
#include "../schedule/message_passing.h"
#include "op_util.h"

namespace tvm {
namespace te {
using namespace arith;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AxisComputeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AxisComputeOpNode*>(node.get());
      p->stream << "axis_compute(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(AxisComputeOpNode);

TVM_REGISTER_GLOBAL("te.AxisComputeOp")
    .set_body_typed([](std::string name, std::string tag, Map<String, ObjectRef> attrs,
                       Array<IterVar> axis,
                       Array<PrimExpr> out_args,
                       Array<PrimExpr> body) { return AxisComputeOp(name, tag, attrs, axis,out_args, body); });


AxisComputeOp::AxisComputeOp(std::string name, std::string tag, Map<String, ObjectRef> attrs,
                             Array<IterVar> axis, Array<PrimExpr> out_args, Array<PrimExpr> body) {
  if (!attrs.defined()) {
    attrs = Map<String, ObjectRef>();
  }
  //   IntSet EvalSet(PrimExpr e, const
  Map<IterVar, IntSet> dom_map;
  for (auto ax : axis) {
    dom_map.Set(ax, IntSet::FromRange(ax->dom));
  }
  Array<IterVar> out_axis;

  for (size_t i = 0; i < out_args.size(); ++i) {
    std::ostringstream os;
    os << "dim_ax" << i;
    IntSet iset = arith::EvalSet(out_args[i], dom_map);
    // IntSet::MatchRange
    auto dom = Range::FromMinExtent(iset.min(), iset.max() + 1 - iset.min());
    out_axis.push_back(IterVar(dom, Var(os.str()), IterVarType::kDataPar));
  }

  auto n = make_object<AxisComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->body = std::move(body);
  n->out_axis = std::move(out_axis);
  n->out_args = std::move(out_args);
  if (n->body[0]->IsInstance<tir::ReduceNode>()) {
    const tir::ReduceNode* reduce = n->body[0].as<tir::ReduceNode>();
    n->reduce_axis = reduce->axis;
  }
  data_ = std::move(n);
}

Array<IterVar> AxisComputeOpNode::output_iter_vars(size_t i) const { return this->out_axis; }

}  // namespace te
}  // namespace tvm