#include "proxsuite-nlp/linesearch-base.hpp"

namespace proxsuite {
namespace nlp {

template class Linesearch<context::Scalar>;
template struct PolynomialTpl<context::Scalar>;
template class ArmijoLinesearch<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
