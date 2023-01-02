/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/problem-base.hpp"

#ifdef PROXNLP_CUSTOM_LDLT
#include "proxnlp/ldlt-allocator.hpp"
#else
#include <Eigen/Cholesky>
#endif

namespace proxnlp {

/** Workspace class, which holds the necessary intermediary data
 * for the solver to function.
 */
template <typename Scalar> struct WorkspaceTpl {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;

  /// Newton iteration variables

  long nx;
  long ndx;
  std::size_t numblocks; // number of constraint blocks
  long numdual;          // total constraint dim

  /// KKT iteration matrix.
  MatrixXs kkt_matrix;
  /// KKT iteration right-hand side.
  VectorXs kkt_rhs;
  /// KKT linear system error (for refinement)
  VectorXs kkt_err;
  /// Primal-dual step \f$\delta w = (\delta x, \delta\lambda)\f$.
  VectorXs pd_step;
  VectorRef prim_step;
  VectorRef dual_step;
  /// Signature of the KKT matrix
  Eigen::VectorXi signature;

  /// LDLT storage
#ifdef PROXNLP_CUSTOM_LDLT
  unique_ptr<linalg::ldlt_base<Scalar>> ldlt_;
#else
  Eigen::LDLT<MatrixXs, Eigen::Lower> ldlt_;
#endif

  //// Data for proximal algorithm

  VectorXs x_prev;
  VectorXs x_trial;
  VectorXs data_lams_prev;
  VectorXs data_lams_trial;
  VectorOfRef lams_prev;
  VectorOfRef lams_trial;

  VectorXs prox_grad;
  MatrixXs prox_hess;

  /// Residuals

  /// Dual residual: gradient of the Lagrangian function
  VectorXs dual_residual;

  VectorXs data_cstr_values;
  /// Values of each constraint
  std::vector<VectorRef> cstr_values;

  VectorXs data_shift_cstr_proj;
  /// Projected values of each constraint
  std::vector<VectorRef> shift_cstr_proj;

  /// Objective value
  Scalar objective_value;
  /// Objective function gradient.
  VectorXs objective_gradient;
  /// Objective function Hessian.
  MatrixXs objective_hessian;
  /// Merit function gradient.
  VectorXs merit_gradient;

  MatrixXs data_jacobians;
  MatrixXs data_hessians;
  MatrixXs data_jacobians_proj;
  std::vector<MatrixRef> cstr_jacobians;
  std::vector<MatrixRef> cstr_vector_hessian_prod;
  std::vector<MatrixRef> cstr_jacobians_proj;

  VectorXs data_shift_cstr_values;
  VectorXs data_lams_plus;
  VectorXs data_lams_pdal;
  VectorXs data_dual_prox_err;

  /// First-order multipliers \f$\mathrm{proj}(\lambda_e + c / \mu)\f$
  std::vector<VectorRef> lams_plus;
  /// Pre-projected multipliers.
  std::vector<VectorRef> shift_cstr_values;
  /// Primal-dual multiplier estimates (from the pdBCL algorithm)
  std::vector<VectorRef> lams_pdal;
  /// Subproblem proximal dual error.
  std::vector<VectorRef> subproblem_dual_err;

  std::vector<Scalar> ls_alphas;
  std::vector<Scalar> ls_values;
  /// Optimal linesearch \f$\alpha^\star\f$
  Scalar alpha_opt;
  /// Merit function derivative in descent direction
  Scalar dmerit_dir = 0.;

  VectorXs tmp_dx_scaled;

  WorkspaceTpl(const Problem &prob, LDLTChoice ldlt_choice = LDLTChoice::DENSE)
      : nx(long(prob.nx())), ndx(long(prob.ndx())),
        numblocks(prob.getNumConstraints()),
        numdual(prob.getTotalConstraintDim()),
        kkt_matrix(ndx + numdual, ndx + numdual), kkt_rhs(ndx + numdual),
        kkt_err(kkt_rhs), pd_step(ndx + numdual), prim_step(pd_step.head(ndx)),
        dual_step(pd_step.tail(numdual)), signature(ndx + numdual),
#ifdef PROXNLP_CUSTOM_LDLT
        ldlt_(allocate_ldlt_from_problem(prob, ldlt_choice)),
#else
        ldlt_(kkt_matrix.cols()),
#endif
        x_prev(nx), x_trial(nx), data_lams_prev(numdual),
        data_lams_trial(numdual), prox_grad(ndx), prox_hess(ndx, ndx),
        dual_residual(ndx), data_cstr_values(numdual),
        data_shift_cstr_proj(numdual), objective_gradient(ndx),
        objective_hessian(ndx, ndx), merit_gradient(ndx),
        data_jacobians(numdual, ndx), data_hessians((int)numblocks * ndx, ndx),
        data_lams_plus(numdual), data_lams_pdal(numdual),
        data_dual_prox_err(numdual), tmp_dx_scaled(ndx) {
    init(prob);
  }

  void init(const Problem &prob) {
    kkt_matrix.setZero();
    kkt_rhs.setZero();
    pd_step.setZero();
    signature.setZero();

    x_prev.setZero();
    x_trial.setZero();
    helpers::allocateMultipliersOrResiduals(prob, data_lams_prev, lams_prev);
    helpers::allocateMultipliersOrResiduals(prob, data_lams_trial, lams_trial);
    prox_grad.setZero();
    prox_hess.setZero();

    dual_residual.setZero();
    helpers::allocateMultipliersOrResiduals(
        prob, data_cstr_values, cstr_values); // not multipliers but same dims
    data_shift_cstr_proj.setZero();
    helpers::allocateMultipliersOrResiduals(prob, data_shift_cstr_proj,
                                            shift_cstr_proj);

    objective_gradient.setZero();
    objective_hessian.setZero();
    merit_gradient.setZero();
    data_jacobians.setZero();
    data_hessians.setZero();

    helpers::allocateMultipliersOrResiduals(prob, data_shift_cstr_values,
                                            shift_cstr_values);
    helpers::allocateMultipliersOrResiduals(prob, data_lams_plus, lams_plus);
    helpers::allocateMultipliersOrResiduals(prob, data_lams_pdal, lams_pdal);
    helpers::allocateMultipliersOrResiduals(prob, data_dual_prox_err,
                                            subproblem_dual_err);
    tmp_dx_scaled.setZero();

    cstr_jacobians.reserve(numblocks);
    cstr_vector_hessian_prod.reserve(numblocks);

    data_jacobians_proj = data_jacobians;

    int cursor = 0;
    int nr = 0;
    for (std::size_t i = 0; i < numblocks; i++) {
      cursor = prob.getIndex(i);
      nr = prob.getConstraintDim(i);
      cstr_jacobians.emplace_back(data_jacobians.middleRows(cursor, nr));
      cstr_jacobians_proj.emplace_back(
          data_jacobians_proj.middleRows(cursor, nr));
      cstr_vector_hessian_prod.emplace_back(
          data_hessians.middleRows((int)i * ndx, ndx));
    }
  }
};

} // namespace proxnlp
