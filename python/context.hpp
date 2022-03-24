#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"

#include <eigenpy/eigen-typedef.hpp>


namespace lienlp {
namespace python {

  namespace context {
    
    using Scalar = double;

    // EIGENPY_MAKE_TYPEDEFS_ALL_SIZES(Scalar, Options, s)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    using Problem_t = Problem<Scalar>;
    using Result_t = SResults<Scalar>;
    using Residual_t = ResidualBase<Scalar>;
    using Cost_t = CostFunctionBase<Scalar>;
    using Constraint_t = ConstraintSetBase<Scalar>;

  } // namespace context

} // namespace python
} // namespace lienlp


