#pragma once

#include "proxsuite-nlp/modelling/constraints/equality-constraint.hpp"
#include "proxsuite-nlp/modelling/constraints/negative-orthant.hpp"
#include "proxsuite-nlp/modelling/constraints/box-constraint.hpp"
#include "proxsuite-nlp/modelling/constraints/l1-penalty.hpp"

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/context.hpp"

namespace proxnlp {

extern template struct ConstraintSetBase<context::Scalar>;
extern template struct ConstraintObjectTpl<context::Scalar>;
extern template struct EqualityConstraint<context::Scalar>;
extern template struct NegativeOrthant<context::Scalar>;
extern template struct BoxConstraintTpl<context::Scalar>;
extern template struct NonsmoothPenaltyL1Tpl<context::Scalar>;

} // namespace proxnlp
#endif
