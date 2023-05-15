/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/cost-function.hpp"

namespace proxnlp {

/// @brief    Defines the sum of one or more cost functions \f$c_1 + c_2 +
/// \cdots\f$
template <typename _Scalar> struct CostSumTpl : CostFunctionBaseTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostFunctionBaseTpl<Scalar>;
  using BasePtr = shared_ptr<Base>;

  std::vector<BasePtr> components_; /// component sub-costs
  std::vector<Scalar> weights_;     /// cost component weights

  CostSumTpl(int nx, int ndx) : Base(nx, ndx) {}

  /// Constructor with a predefined vector of components.
  CostSumTpl(int nx, int ndx, const std::vector<BasePtr> &comps,
             const std::vector<Scalar> &weights)
      : Base(nx, ndx), components_(comps), weights_(weights) {
    assert(components_.size() == weights_.size());
  }

  std::size_t numComponents() const { return components_.size(); }

  auto clone() const { return std::make_shared<CostSumTpl>(*this); }

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

  void addComponent(shared_ptr<Base> comp, const Scalar w = 1.) {
    components_.push_back(comp);
    weights_.push_back(w);
  }

  CostSumTpl<Scalar> &operator+=(const shared_ptr<Base> &other) {
    addComponent(other);
    return *this;
  }

  CostSumTpl<Scalar> &operator+=(const CostSumTpl<Scalar> &other) {
    components_.insert(components_.end(), other.components_.begin(),
                       other.components_.end());
    weights_.insert(weights_.end(), other.weights_.begin(),
                    other.weights_.end());
    return *this;
  }

  CostSumTpl<Scalar> &operator*=(Scalar rhs) {
    for (auto &weight : weights_) {
      weight = rhs * weight;
    }
    return *this;
  }

  // printing
  friend std::ostream &operator<<(std::ostream &ostr,
                                  const CostSumTpl<Scalar> &cost) {
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

  friend auto operator*(CostSumTpl const &self, Scalar a)
      -> shared_ptr<CostSumTpl> {
    auto out = self.clone();
    (*out) *= a;
    return out;
  }

  friend auto operator*(Scalar a, CostSumTpl const &self) { return self * a; }

  friend auto operator-(CostSumTpl const &self) {
    return self * static_cast<Scalar>(-1.);
  }
};

} // namespace proxnlp

#include "proxnlp/cost-sum.hxx"

#ifdef PROXNLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxnlp/cost-sum.txx"
#endif
