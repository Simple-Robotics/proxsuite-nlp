#include "proxnlp/function-base.hpp"
#include "proxnlp/function-ops.hpp"
#include "proxnlp/cost-function.hpp"
#include "proxnlp/manifold-base.hpp"

namespace proxnlp {

template struct BaseFunctionTpl<context::Scalar>;

template struct C1FunctionTpl<context::Scalar>;

template struct C2FunctionTpl<context::Scalar>;

template struct ComposeFunctionTpl<context::Scalar>;

} // namespace proxnlp
