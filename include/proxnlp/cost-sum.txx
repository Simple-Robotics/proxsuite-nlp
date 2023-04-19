#pragma once

#include "./cost-sum.hpp"
#include "proxnlp/context.hpp"

namespace proxnlp {

extern template struct CostSumTpl<context::Scalar>;

}
