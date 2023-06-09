#include "proxnlp/modelling/constraints.hpp"

namespace proxnlp {

template struct ConstraintSetBase<context::Scalar>;
template struct EqualityConstraint<context::Scalar>;
template struct NegativeOrthant<context::Scalar>;
template struct BoxConstraintTpl<context::Scalar>;

} // namespace proxnlp
