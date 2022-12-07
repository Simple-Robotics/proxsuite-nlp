/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
/// @brief     Basis for merit functions.
#pragma once

#include "proxnlp/problem-base.hpp"

namespace proxnlp {

template <typename Scalar, typename... Args> struct MeritFunctionBaseTpl {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;

  shared_ptr<Problem> problem_;

  MeritFunctionBaseTpl(shared_ptr<Problem> prob) : problem_(prob) {}

  /// Evaluate the merit function.
  virtual Scalar evaluate(const ConstVectorRef &x,
                          const Args &...args) const = 0;
  /// Evaluate the merit function gradient.
  virtual void computeGradient(const ConstVectorRef &x, const Args &...args,
                               VectorRef out) const = 0;
};

} // namespace proxnlp
