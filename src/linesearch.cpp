#include "proxsuite-nlp/linesearch-base.hpp"

namespace proxnlp {

template class Linesearch<context::Scalar>;
template struct PolynomialTpl<context::Scalar>;
template class ArmijoLinesearch<context::Scalar>;

} // namespace proxnlp
