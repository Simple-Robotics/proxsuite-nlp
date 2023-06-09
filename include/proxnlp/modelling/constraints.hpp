#pragma once

#include "proxnlp/modelling/constraints/equality-constraint.hpp"
#include "proxnlp/modelling/constraints/negative-orthant.hpp"
#include "proxnlp/modelling/constraints/box-constraint.hpp"

#ifdef PROXNLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxnlp/context.hpp"

namespace proxnlp {

extern template struct ConstraintSetBase<context::Scalar>;
extern template struct EqualityConstraint<context::Scalar>;
extern template struct NegativeOrthant<context::Scalar>;
extern template struct BoxConstraintTpl<context::Scalar>;

} // namespace proxnlp
#endif
