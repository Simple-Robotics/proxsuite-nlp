/** Copyright (c) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"


namespace lienlp {

  template<typename _Scalar>
  struct SResults
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

    bool converged = false;

    VectorXs xOpt;
    VectorOfVectors lamsOpt;

    /// Final solver parameters
    Scalar mu;
    Scalar rho;

    SResults(const int nx,
             const Prob_t& prob)
             : xOpt(nx),
               mu(0.),
               rho(0.)
    {
      Prob_t::allocateMultipliers(prob, lamsOpt);
    }

  };

} // namespace lienlp

