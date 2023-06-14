#pragma once

#include "proxnlp/cost-function.hpp"

namespace proxnlp {

extern template struct CostFunctionBaseTpl<context::Scalar>;

extern template
auto downcast_function_to_cost<context::Scalar>(const shared_ptr<context::C2Function>&) -> shared_ptr<context::Cost>;

}
