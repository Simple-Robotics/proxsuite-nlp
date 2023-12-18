#pragma once

#include "./cost-sum.hpp"
#include "proxsuite-nlp/context.hpp"

namespace proxnlp {

extern template struct CostSumTpl<context::Scalar>;

}
