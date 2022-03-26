/** Copyright (c) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"


namespace lienlp
{

  enum ConvergedFlag
  {
    UNINIT=-1,
    SUCCESS=0,
    TOO_MANY_ITERS=1
  };

  /**
   * @brief   Results struct, holding the returned data from the solver.
   * 
   * @details This struct holds the current (and output) primal-dual point,
   *          the optimal proximal parameters \f$(\rho, \mu)\f$.
   */
  template<typename _Scalar>
  struct SResults
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;
    using VecBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    ConvergedFlag converged = ConvergedFlag::UNINIT;

    Scalar value;
    VectorXs xOpt;
    VectorOfVectors lamsOpt;
    std::vector<VecBool> activeSet;

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
      Prob_t::allocateMultipliersOrResiduals(prob, lamsOpt);
      activeSet.reserve(prob.getNumConstraints());
      for (std::size_t i = 0; i < prob.getNumConstraints(); i++)
      {
        activeSet.push_back(VecBool::Zero(prob.getConstraint(i)->nr()));
      }
    }

    friend std::ostream& operator<<(std::ostream& s, const SResults<Scalar>& self)
    {
      s << "{\n"
        << "  convergence:  " << self.converged << ",\n"
        << "  value:        " << self.value << ",\n"
        << "  numIters:     " << self.numIters << ",\n"
        << "  mu:           " << self.mu << ",\n"
        << "  rho:          " << self.rho << ",\n";
      for (std::size_t i = 0; i < self.activeSet.size(); i++)
      {
        s << "  activeSet[" << i << "]: "
          << self.activeSet[i].transpose() << ",\n";
      }
      s << "}";
      return s;
    }


  };

} // namespace lienlp

