/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/problem-base.hpp"

#include <Eigen/Cholesky>

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

  int nx;
  int ndx;
  std::size_t numblocks; // number of constraint blocks
  int numdual;           // total constraint dim

  /// KKT iteration matrix.
  MatrixXs kkt_matrix;
  /// KKT iteration right-hand side.
  VectorXs kkt_rhs;
  /// Primal-dual step \f$\delta w = (\delta x, \delta\lambda)\f$.
  VectorXs pd_step;
  VectorRef prim_step;
  VectorRef dual_step;
  /// Signature of the matrix
  Eigen::VectorXi signature;

  /// LDLT storage
  Eigen::LDLT<MatrixXs, Eigen::Lower> ldlt_;

  //// Data for proximal algorithm

  VectorXs x_prev;
  VectorXs x_trial;
  VectorXs data_lams_prev;
  VectorXs lams_trial_data;
  VectorOfRef lams_prev;
  VectorOfRef lams_trial;

  VectorXs prox_grad;
  MatrixXs prox_hess;

  /// Residuals

  /// Dual residual: gradient of the Lagrangian function
  VectorXs dual_residual;
  VectorXs data_cstr_values;
  /// Values of each constraint
  VectorOfRef cstr_values;
  VectorXs data_cstr_values_proj;
  /// Projected values of each constraint
  std::vector<VectorRef> cstr_values_proj;

  /// Objective value
  Scalar objective_value;
  /// Objective function gradient.
  VectorXs objective_gradient;
  /// Objective function Hessian.
  MatrixXs objective_hessian;
  /// Merit function gradient.
  VectorXs merit_gradient;

  MatrixXs jacobians_data;
  MatrixXs hessians_data;
  std::vector<MatrixRef> cstr_jacobians;
  std::vector<MatrixRef> cstr_vector_hessian_prod;
  MatrixXs jacobians_proj_data;
  std::vector<MatrixRef> cstr_jacobians_proj;

  VectorXs data_shift_cstr_values;
  VectorXs data_lams_plus;
  VectorXs data_lams_pdal;
  VectorXs data_dual_prox_err;

  /// First-order multipliers \f$\mathrm{proj}(\lambda_e + c / \mu)\f$
  VectorOfRef lams_plus;
  /// Pre-projected multipliers.
  VectorOfRef shift_cstr_values;
  /// Primal-dual multiplier estimates (from the pdBCL algorithm)
  VectorOfRef lams_pdal;
  /// Subproblem proximal dual error.
  VectorOfRef subproblemDualErr;

  std::vector<Scalar> ls_alphas;
  std::vector<Scalar> ls_values;
  Scalar alpha_opt;
  /// Merit function derivative in descent direction
  Scalar dmerit_dir = 0.;

  WorkspaceTpl(const Problem &prob)
      : nx(prob.nx()), ndx(prob.ndx()), numblocks(prob.getNumConstraints()),
        numdual(prob.getTotalConstraintDim()),
        kkt_matrix(ndx + numdual, ndx + numdual), kkt_rhs(ndx + numdual),
        pd_step(ndx + numdual), prim_step(pd_step.head(ndx)),
        dual_step(pd_step.tail(numdual)), signature(ndx + numdual),
        ldlt_(kkt_matrix), x_prev(nx), x_trial(nx), data_lams_prev(numdual),
        lams_trial_data(numdual), prox_grad(ndx), prox_hess(ndx, ndx),
        dual_residual(ndx), data_cstr_values(numdual),
        data_cstr_values_proj(numdual), objective_gradient(ndx),
        objective_hessian(ndx, ndx), merit_gradient(ndx),
        jacobians_data(numdual, ndx), hessians_data((int)numblocks * ndx, ndx),
        data_lams_plus(numdual), data_lams_pdal(numdual),
        data_dual_prox_err(numdual) {
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
    helpers::allocateMultipliersOrResiduals(prob, lams_trial_data, lams_trial);
    prox_grad.setZero();
    prox_hess.setZero();

    dual_residual.setZero();
    helpers::allocateMultipliersOrResiduals(
        prob, data_cstr_values, cstr_values); // not multipliers but same dims
    data_cstr_values_proj.setZero();
    helpers::allocateMultipliersOrResiduals(prob, data_cstr_values_proj,
                                            cstr_values_proj);

    objective_gradient.setZero();
    objective_hessian.setZero();
    merit_gradient.setZero();
    jacobians_data.setZero();
    hessians_data.setZero();

    helpers::allocateMultipliersOrResiduals(prob, data_shift_cstr_values,
                                            shift_cstr_values);
    helpers::allocateMultipliersOrResiduals(prob, data_lams_plus, lams_plus);
    helpers::allocateMultipliersOrResiduals(prob, data_lams_pdal, lams_pdal);
    helpers::allocateMultipliersOrResiduals(prob, data_dual_prox_err,
                                            subproblemDualErr);

    cstr_jacobians.reserve(numblocks);
    cstr_vector_hessian_prod.reserve(numblocks);

    jacobians_proj_data = jacobians_data;

    int cursor = 0;
    int nr = 0;
    for (std::size_t i = 0; i < numblocks; i++) {
      cursor = prob.getIndex(i);
      nr = prob.getConstraintDim(i);
      cstr_jacobians.emplace_back(jacobians_data.middleRows(cursor, nr));
      cstr_jacobians_proj.emplace_back(
          jacobians_proj_data.middleRows(cursor, nr));
      cstr_vector_hessian_prod.emplace_back(
          hessians_data.middleRows((int)i * ndx, ndx));
    }
  }
};

} // namespace proxnlp
