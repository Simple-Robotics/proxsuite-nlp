#include "proxsuite-nlp/function-base.hpp"
#include "proxsuite-nlp/function-ops.hpp"
#include "proxsuite-nlp/cost-function.hpp"
#include "proxsuite-nlp/manifold-base.hpp"

namespace proxsuite {
namespace nlp {

template struct BaseFunctionTpl<context::Scalar>;

template struct C1FunctionTpl<context::Scalar>;

template struct C2FunctionTpl<context::Scalar>;

template struct ComposeFunctionTpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
