#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/function-base.hpp"

namespace proxnlp {

extern template struct BaseFunctionTpl<context::Scalar>;

extern template struct C1FunctionTpl<context::Scalar>;

extern template struct C2FunctionTpl<context::Scalar>;

} // namespace proxnlp
