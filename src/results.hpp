/** Copyright (c) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include "lienlp/macros.hpp"


namespace lienlp {

  template<typename _Scalar>
  struct SResult
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    VectorXs x_opt;
    VectorOfVectors lams_opt;

    bool converged;

    /// Final solver parameters
    Scalar mu;

  }

} // namespace lienlp

