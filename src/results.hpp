/** Copyright (c) 2022 LAAS-CNRS, INRIA
 * 
 */
#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/problem-base.hpp"


namespace lienlp
{

  enum ConvergenceFlag
  {
    UNINIT=-1,
    SUCCESS=0,
    MAX_ITERS_REACHED=1
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
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using Problem = ProblemTpl<Scalar>;
    using VecBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    ConvergenceFlag converged = ConvergenceFlag::UNINIT;

    Scalar merit;
    Scalar value;
    VectorXs xOpt;
    VectorXs lamsOpt_data;
    VectorOfRef lamsOpt;
    std::vector<VecBool> activeSet;
    Scalar dualInfeas = 0.;
    Scalar primalInfeas = 0.;

    /// Final solver parameters
    std::size_t numIters = 0;
    Scalar mu;
    Scalar rho;

    SResults(const int nx, const Problem& prob)
             : xOpt(nx)
             , lamsOpt_data(prob.getTotalConstraintDim())
             , numIters(0)
             , mu(0.)
             , rho(0.)
    {
      xOpt.setZero();
      helpers::allocateMultipliersOrResiduals(prob, lamsOpt_data, lamsOpt);
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
        << "  merit:        " << self.merit << ",\n"
        << "  value:        " << self.value << ",\n"
        << "  numIters:     " << self.numIters << ",\n"
        << "  mu:           " << self.mu << ",\n"
        << "  rho:          " << self.rho << ",\n"
        << "  dual_infeas   " << self.dualInfeas << ",\n"
        << "  primal_infeas " << self.primalInfeas << ",\n";
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

