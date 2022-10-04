#pragma once

#include "proxnlp/cost-function.hpp"

namespace proxnlp {

/// @brief    Defines the sum of one or more cost functions \f$c_1 + c_2 +
/// \cdots\f$
template <typename _Scalar> struct CostSum : CostFunctionBaseTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostFunctionBaseTpl<Scalar>;
  using BasePtr = Base const *;

  std::vector<BasePtr> components_; /// component sub-costs
  std::vector<Scalar> weights_;     /// cost component weights

  CostSum(int nx, int ndx) : Base(nx, ndx) {}

  /// Constructor with a predefined vector of components.
  CostSum(int nx, int ndx, const std::vector<BasePtr> &comps,
          const std::vector<Scalar> &weights)
      : Base(nx, ndx), components_(comps), weights_(weights) {
    assert(components_.size() == weights_.size());
  }

  std::size_t numComponents() const { return components_.size(); }

  Scalar call(const ConstVectorRef &x) const {
    Scalar result_ = 0.;
    for (std::size_t i = 0; i < numComponents(); i++) {
      result_ += weights_[i] * components_[i]->call(x);
    }
    return result_;
  }

  void computeGradient(const ConstVectorRef &x, VectorRef out) const {
    out.setZero();
    for (std::size_t i = 0; i < numComponents(); i++) {
      out.noalias() = out + weights_[i] * components_[i]->computeGradient(x);
    }
  }

  void computeHessian(const ConstVectorRef &x, MatrixRef out) const {
    out.setZero();
    for (std::size_t i = 0; i < numComponents(); i++) {
      out.noalias() = out + weights_[i] * components_[i]->computeHessian(x);
    }
  }

  /* CostSum API definition */

  void addComponent(const Base &comp, const Scalar w = 1.) {
    components_.push_back(&comp);
    weights_.push_back(w);
  }

  CostSum<Scalar> &operator+=(const Base &other) {
    addComponent(other);
    return *this;
  }

  CostSum<Scalar> &operator+=(const CostSum<Scalar> &other) {
    components_.insert(components_.end(), other.components_.begin(),
                       other.components_.end());
    weights_.insert(weights_.end(), other.weights_.begin(),
                    other.weights_.end());
    return *this;
  }

  // increment this using an rvalue (for ex tmp rhs)
  CostSum<Scalar> &operator+=(CostSum<Scalar> &&rhs) {
    (*this) += std::forward<CostSum<Scalar>>(rhs);
    return *this;
  }

  CostSum<Scalar> &operator*=(const Scalar rhs) {
    for (auto &weight : weights_) {
      weight = rhs * weight;
    }
    return *this;
  }

  // printing
  friend std::ostream &operator<<(std::ostream &ostr,
                                  const CostSum<Scalar> &cost) {
    const std::size_t nc = cost.numComponents();
    ostr << "CostSum(num_components=" << nc;
    ostr << ", weights=(";
    for (std::size_t i = 0; i < nc; i++) {
      ostr << cost.weights_[i];
      if (i < nc - 1)
        ostr << ", ";
    }
    ostr << ")";
    ostr << ")";
    return ostr;
  }
};

} // namespace proxnlp

#include "proxnlp/cost-sum.hxx"
