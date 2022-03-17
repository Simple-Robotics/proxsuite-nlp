/** Copyright (c) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"


namespace lienlp {

  enum ConvergedFlag
  {
    UNINIT=-1,
    SUCCESS=0,
    TOO_MANY_ITERS=1
  };

  template<typename _Scalar>
  struct SResults
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

    ConvergedFlag converged = ConvergedFlag::UNINIT;

    Scalar value;
    VectorXs xOpt;
    VectorOfVectors lamsOpt;

    /// Final solver parameters
    std::size_t numIters = 0;
    Scalar mu;
    Scalar rho;

    SResults(const int nx,
             const Prob_t& prob)
             : xOpt(nx),
               numIters(0),
               mu(0.),
               rho(0.)
    {
      Prob_t::allocateMultipliers(prob, lamsOpt);
    }

    friend std::ostream& operator<<(std::ostream& s, const SResults<Scalar>& self)
    {
      s << "{\n"
        << "  convergence: " << self.converged << ",\n"
        << "  value:       " << self.value << ",\n"
        << "  numIters:    " << self.numIters << ",\n"
        << "  mu:          " << self.mu << ",\n"
        << "}";
    }


  };

} // namespace lienlp

