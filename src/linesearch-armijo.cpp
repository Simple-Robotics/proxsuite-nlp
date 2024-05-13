#include "proxsuite-nlp/linesearch-armijo.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    PolynomialTpl<context::Scalar>;
template class PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ArmijoLinesearch<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
