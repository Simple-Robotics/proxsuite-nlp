/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/manifold-base.hpp"
#include "proxsuite-nlp/cost-function.hpp"
#include "proxsuite-nlp/constraint-base.hpp"

namespace proxsuite {
namespace nlp {

template <typename _Scalar> struct ProblemTpl {
public:
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

  /// Generic constraint type
  using ConstraintObject = ConstraintObjectTpl<Scalar>;
  using ConstraintPtr = shared_ptr<ConstraintObject>;
  /// Cost function type
  using CostType = CostFunctionBaseTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;

  /// The working manifold \f$M\f$.
  polymorphic<Manifold> manifold_;
  /// The cost function.
  shared_ptr<CostType> cost_;
  /// The set of constraints.
  std::vector<ConstraintObject> constraints_;

  const CostType &cost() const { return *cost_; }
  const Manifold &manifold() const { return *manifold_; }

  template <class U>
  ProblemTpl(U &&manifold, shared_ptr<CostType> cost,
             const std::vector<ConstraintObject> &constraints = {})
      : manifold_(std::forward<U>(manifold)), cost_(cost),
        constraints_(constraints), nc_total_(0) {
    reset_constraint_dim_vars();
  }

  /// Get a pointer to the \f$i\f$-th constraint pointer
  const ConstraintObject &getConstraint(const std::size_t &i) const {
    return constraints_[i];
  }

  /// @brief Get the number of constraint blocks.
  std::size_t getNumConstraints() const { return constraints_.size(); }

  int getTotalConstraintDim() const { return nc_total_; }

  /// Get dimension of constraint \p i.
  int getConstraintDim(std::size_t i) const { return ncs_[i]; }

  int nx() const { return manifold_->nx(); }
  int ndx() const { return manifold_->ndx(); }

  /// @brief Add a constraint to the problem, after initialization.
  template <typename T> void addConstraint(T &&cstr) {
    constraints_.push_back(std::forward<T>(cstr));
    reset_constraint_dim_vars();
  }

  auto getSegment(VectorXs &x, std::size_t i) const {
    return x.segment(getIndex(i), getConstraintDim(i));
  }

  auto getConstSegment(const VectorXs &x, std::size_t i) const {
    return x.segment(getIndex(i), getConstraintDim(i));
  }

  std::vector<int> getIndices() const { return indices_; }

  int getIndex(std::size_t i) const { return indices_[i]; }

  void evaluate(const ConstVectorRef &x, Workspace &workspace) const {
    workspace.objective_value = cost().call(x);

    for (std::size_t i = 0; i < getNumConstraints(); i++) {
      const ConstraintObject &cstr = constraints_[i];
      workspace.cstr_values[i] = cstr.func()(x);
    }
  }

  void computeDerivatives(const ConstVectorRef &x, Workspace &workspace) const {
    cost().computeGradient(x, workspace.objective_gradient);

    for (std::size_t i = 0; i < getNumConstraints(); i++) {
      const ConstraintObject &cstr = constraints_[i];
      cstr.func().computeJacobian(x, workspace.cstr_jacobians[i]);
    }
  }

  void computeHessians(const ConstVectorRef &x, Workspace &workspace,
                       bool evaluate_all_constraint_hessians = false) const {
    cost().computeHessian(x, workspace.objective_hessian);
    for (std::size_t i = 0; i < getNumConstraints(); i++) {
      const ConstraintObject &cstr = getConstraint(i);
      bool use_vhp =
          !cstr.set_->disableGaussNewton() || evaluate_all_constraint_hessians;
      if (use_vhp)
        cstr.func().vectorHessianProduct(x, workspace.lams_pdal[i],
                                         workspace.cstr_vector_hessian_prod[i]);
    }
  }

protected:
  /// Total number of constraints
  int nc_total_;
  std::vector<int> ncs_;
  std::vector<int> indices_;

  /// Set values of const data members for constraint dimensions
  void reset_constraint_dim_vars() {
    ncs_.clear();
    indices_.clear();
    int cursor = 0;
    int nr = 0;
    for (std::size_t i = 0; i < constraints_.size(); i++) {
      const ConstraintObject &cstr = constraints_[i];
      nr = cstr.func().nr();
      ncs_.push_back(nr);
      indices_.push_back(cursor);
      cursor += nr;
    }
    nc_total_ = cursor;
  }
};

namespace helpers {

template <typename Scalar, typename VectorType>
void createConstraintWiseView(const ProblemTpl<Scalar> &prob,
                              typename math_types<Scalar>::VectorXs &input,
                              std::vector<Eigen::Ref<VectorType>> &out) {
  static_assert(VectorType::IsVectorAtCompileTime,
                "Function only supports compile-time vectors.");
  out.reserve(prob.getNumConstraints());
  for (std::size_t i = 0; i < prob.getNumConstraints(); i++) {
    out.emplace_back(prob.getSegment(input, i));
  }
}

/// @brief   Allocate a set of multipliers (or residuals) for a given problem
/// instance.
template <typename Scalar>
void allocateMultipliersOrResiduals(
    const ProblemTpl<Scalar> &prob, typename math_types<Scalar>::VectorXs &data,
    typename math_types<Scalar>::VectorOfRef &out) {
  data.resize(prob.getTotalConstraintDim());
  data.setZero();
  createConstraintWiseView(prob, data, out);
}

} // namespace helpers

} // namespace nlp
} // namespace proxsuite

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/problem-base.txx"
#endif
