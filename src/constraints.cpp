#include "proxsuite-nlp/modelling/constraints.hpp"

namespace proxsuite {
namespace nlp {

template struct ConstraintSetBase<context::Scalar>;
template struct ConstraintObjectTpl<context::Scalar>;
template struct EqualityConstraint<context::Scalar>;
template struct NegativeOrthant<context::Scalar>;
template struct BoxConstraintTpl<context::Scalar>;
template struct NonsmoothPenaltyL1Tpl<context::Scalar>;

} // namespace nlp
} // namespace proxsuite
