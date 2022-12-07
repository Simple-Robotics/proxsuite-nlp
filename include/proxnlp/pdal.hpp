/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/merit-function-base.hpp"

namespace proxnlp {

/**
 * @brief   Primal-dual augmented Lagrangian-type merit function.
 *
 * Primal-dual Augmented Lagrangian function, extending
 * the function from Gill & Robinson (2012) to inequality constraints.
 * For inequality constraints of the form \f$ c(x) \in \calC \f$ and an
 * objective function \f$ f\colon\calX \to \RR \f$, \f[ \calM_{\mu}(x, \lambda;
 * \lambda_e) = f(x) + \frac{1}{2\mu} \| \proj_\calC(c(x) + \mu \lambda_e)
 * \|_2^2
 *    + \frac{1}{2\mu} \| \proj_\calC(c(x) + \mu\lambda_e) - \mu\lambda) \|_2^2.
 * \f]
 *
 */
template <typename _Scalar> struct PDALFunction {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;

  shared_ptr<Problem> problem_;

protected:
  /// AL penalty parameter
  Scalar mu_;
  /// Reciprocal penalty parameter
  Scalar mu_inv_ = 1. / mu_;

public:
  /// Generalized pdAL dual penalty param
  Scalar gamma_;

  PDALFunction(shared_ptr<Problem> prob, const Scalar mu, const Scalar gamma);

  Scalar evaluate(const ConstVectorRef &x, const VectorOfRef &lams,
                  const std::vector<VectorRef> &shift_cvals,
                  const std::vector<VectorRef> &proj_cvals) const;

  /// @brief  Set the merit function penalty parameter.
  void setPenalty(const Scalar &new_mu) noexcept;
};

} // namespace proxnlp

#include "proxnlp/pdal.hxx"
