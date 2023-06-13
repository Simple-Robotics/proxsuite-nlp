#pragma once

#include "proxnlp/context.hpp"
#include "proxnlp/linesearch-base.hpp"

namespace proxnlp {

extern template class Linesearch<context::Scalar>;
extern template struct PolynomialTpl<context::Scalar>;
extern template class ArmijoLinesearch<context::Scalar>;

} // namespace proxnlp
