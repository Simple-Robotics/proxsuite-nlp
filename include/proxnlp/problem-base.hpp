#pragma once

#include "proxnlp/manifold-base.hpp"
#include "proxnlp/cost-function.hpp"
#include "proxnlp/constraint-base.hpp"
#include "proxnlp/modelling/constraints/equality-constraint.hpp"

#include <vector>

namespace proxnlp {

template <typename _Scalar> struct ProblemTpl {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  /// Generic constraint type
  using ConstraintType = ConstraintObject<Scalar>;
  using ConstraintPtr = shared_ptr<ConstraintType>;
  /// Equality constraint type
  using EqualityType = EqualityConstraint<Scalar>;
  /// Cost function type
  using CostType = CostFunctionBaseTpl<Scalar>;

  /// The cost functional
  const CostType &cost_;

  /// Get a pointer to the \f$i\f$-th constraint pointer
  const ConstraintType &getConstraint(const std::size_t &i) const {
    return constraints_[i];
  }

  /// @brief Get the number of constraint blocks.
  std::size_t getNumConstraints() const { return constraints_.size(); }

  int getTotalConstraintDim() const { return m_nc_total; }

  /// Get dimension of constraint \p i.
  int getConstraintDim(std::size_t i) const { return m_ncs[i]; }

  std::size_t nx() const { return cost_.nx(); }
  std::size_t ndx() const { return cost_.ndx(); }

  ProblemTpl(const CostType &cost) : cost_(cost), m_nc_total(0) {
    reset_constraint_dim_vars();
  }

  ProblemTpl(const CostType &cost,
             const std::vector<ConstraintType> &constraints)
      : cost_(cost), constraints_(constraints), m_nc_total(0) {
    reset_constraint_dim_vars();
  }

  /// @brief Add a constraint to the problem, after initialization.
  template <typename T> void addConstraint(T &&cstr) {
    constraints_.push_back(std::forward<T>(cstr));
    reset_constraint_dim_vars();
  }

  std::vector<int> getIndices() const { return m_indices; }

  int getIndex(std::size_t i) const { return m_indices[i]; }

  void evaluate(const ConstVectorRef &x,
                WorkspaceTpl<Scalar> &workspace) const {
    workspace.objectiveValue = cost_.call(x);
    for (std::size_t i = 0; i < getNumConstraints(); i++) {
      const ConstraintType &cstr = constraints_[i];
      workspace.cstrValues[i] = cstr.func()(x);
    }
  }

  void computeDerivatives(const ConstVectorRef &x,
                          WorkspaceTpl<Scalar> &workspace) const {
    cost_.computeGradient(x, workspace.objectiveGradient);
    for (std::size_t i = 0; i < getNumConstraints(); i++) {
      const ConstraintType &cstr = constraints_[i];
      cstr.func().computeJacobian(x, workspace.cstrJacobians[i]);
    }
  }

protected:
  /// Vector of constraints.
  std::vector<ConstraintType> constraints_;
  /// Total number of constraints
  int m_nc_total;
  std::vector<int> m_ncs;
  std::vector<int> m_indices;

  /// Set values of const data members for constraint dimensions
  void reset_constraint_dim_vars() {
    m_ncs.clear();
    m_indices.clear();
    int cursor = 0;
    int nr = 0;
    for (std::size_t i = 0; i < constraints_.size(); i++) {
      const ConstraintType &cstr = constraints_[i];
      nr = cstr.func().nr();
      m_ncs.push_back(nr);
      m_indices.push_back(cursor);
      cursor += nr;
    }
    m_nc_total = cursor;
  }
};

namespace helpers {
/// @brief   Allocate a set of multipliers (or residuals) for a given problem
/// instance.
template <typename Scalar>
void allocateMultipliersOrResiduals(
    const ProblemTpl<Scalar> &prob, typename math_types<Scalar>::VectorXs &data,
    typename math_types<Scalar>::VectorOfRef &out) {
  data.resize(prob.getTotalConstraintDim());
  data.setZero();
  out.reserve(prob.getNumConstraints());
  int cursor = 0;
  int nr = 0;
  for (std::size_t i = 0; i < prob.getNumConstraints(); i++) {
    const ConstraintObject<Scalar> &cstr = prob.getConstraint(i);
    nr = cstr.func().nr();
    out.emplace_back(data.segment(cursor, nr));
    cursor += nr;
  }
}

} // namespace helpers

} // namespace proxnlp
