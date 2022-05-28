/// @file solver-base.hxx
/// Implementations for the prox solver.
#pragma once

#include "proxnlp/solver-base.hpp"

#include <stdexcept>

namespace proxnlp
{
  template<typename Scalar>
  ConvergenceFlag SolverTpl<Scalar>::
  solve(Workspace& workspace,
        Results& results,
        const ConstVectorRef& x0,
        const std::vector<VectorRef>& lams0)
  {
    VectorXs new_lam(problem->getTotalConstraintDim());
    new_lam.setZero();
    int nr = 0;
    const std::size_t numc = problem->getNumConstraints();
    if (numc != lams0.size())
    {
      throw std::runtime_error("Specified number of constraints is not the same "
                                "as the provided number of multipliers!");
    }
    for (std::size_t i = 0; i < numc; i++)
    {
      nr = problem->getConstraintDims()[i];
      new_lam.segment(problem->getIndex(i), nr) = lams0[i];
    }
    return solve(workspace, results, x0, new_lam);
  }

  template<typename Scalar>
  typename SolverTpl<Scalar>::InertiaFlag SolverTpl<Scalar>::checkInertia(const Eigen::VectorXi& signature) const
  {
    const int ndx = manifold.ndx();
    const int numc = problem->getTotalConstraintDim();
    const long n = signature.size();
    int numpos = 0;
    int numneg = 0;
    int numzer = 0;
    for (long i = 0; i < n; i++)
    {
      switch (signature(i))
      {
      case 1 : numpos++;
                break;
      case 0 : numzer++;
                break;
      case -1: numneg++;
                break;
      default: throw std::runtime_error("Matrix signature should only have Os, 1s, and -1s.");
      }
    }
    InertiaFlag flag = OK;
    bool print_info = verbose >= 2;
    if (print_info) fmt::print(" | Inertia: {:d}+, {:d}, {:d}-", numpos, numzer, numneg);
    bool pos_ok = numpos == ndx;
    bool neg_ok = numneg == numc;
    bool zer_ok = numzer == 0;
    if (!(pos_ok && neg_ok && zer_ok))
    {
      if (print_info) fmt::print(" is incorrect");
      if (!zer_ok) flag = ZEROS;
      else flag = BAD;
    } else {
      if (print_info) fmt::print(fmt::fg(fmt::color::pale_green), " OK");
    }
    return flag;
  }

  template<typename S>
  void SolverTpl<S>::computeResidualsAndMultipliers(
    const ConstVectorRef& x,
    const ConstVectorRef& lams_data,
    Workspace& workspace
  ) const
  {
    problem->evaluate(x, workspace);
    workspace.lamsPlusPre_data = workspace.lamsPrev_data + mu_eq_inv_ * workspace.cstr_values_data;
    // project multiplier estimate
    for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
    {
      const typename Problem::ConstraintPtr& cstr = problem->getConstraint(i);
      workspace.lamsPlus[i] = cstr->m_set->normalConeProjection(workspace.lamsPlusPre[i]);
    }
    workspace.dual_prox_err_data = mu_eq_ * (workspace.lamsPlus_data - lams_data);
    workspace.lamsPDAL_data = 2 * workspace.lamsPlus_data - lams_data;
  }

  /// Compute problem derivatives
  template<typename S>
  void SolverTpl<S>::computeResidualDerivatives(
    const ConstVectorRef& x,
    Workspace& workspace,
    bool second_order) const
  {
    problem->computeDerivatives(x, workspace);
    if (second_order)
    {
      problem->m_cost.computeHessian(x, workspace.objectiveHessian);

    }
    for (std::size_t i = 0; i < problem->getNumConstraints(); i++)
    {
      const typename Problem::ConstraintPtr& cstr = problem->getConstraint(i);
      cstr->m_set->applyNormalConeProjectionJacobian(workspace.lamsPlusPre[i], workspace.cstrJacobians[i]);

      bool use_vhp = (use_gauss_newton && !(cstr->m_set->disableGaussNewton())) || !(use_gauss_newton);
      if (second_order && use_vhp)
      {
        cstr->m_func.vectorHessianProduct(x, workspace.lamsPDAL[i], workspace.cstrVectorHessianProd[i]);
      }
    }
    if (rho_ > 0.)
    {
      prox_penalty.computeGradient(x, workspace.prox_grad);
      if (second_order)
        prox_penalty.computeHessian(x, workspace.prox_hess);
    }
  } 


} // namespace proxnlp
