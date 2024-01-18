#include "proxsuite-nlp/linesearch-base.hpp"

namespace proxsuite {
namespace nlp {

template class PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    Linesearch<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    PolynomialTpl<context::Scalar>;
template class PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ArmijoLinesearch<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
