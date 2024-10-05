#include "proxsuite-nlp/constraint-set.hpp"

namespace proxsuite::nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ConstraintSetTpl<context::Scalar>;
template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ConstraintObjectTpl<context::Scalar>;

} // namespace proxsuite::nlp
