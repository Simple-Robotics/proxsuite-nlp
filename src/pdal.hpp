#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/lagrangian.hpp"

#include "proxnlp/workspace.hpp"

#include <vector>

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
  using LagrangianType = LagrangianFunction<Scalar>;

  shared_ptr<Problem> problem_;
  LagrangianType lagrangian_;
  /// Generalized pdAL dual penalty param
  const Scalar gamma_ = 1.;

  PDALFunction(shared_ptr<Problem> prob, const Scalar mu);

  /**
   *  @brief Compute the first-order multiplier estimates.
   */
  void computeFirstOrderMultipliers(const ConstVectorRef &x,
                                    const VectorOfRef &lams_ext,
                                    std::vector<VectorRef> &lams_cache,
                                    std::vector<VectorRef> &out) const;

  Scalar evaluate(const ConstVectorRef &x, const VectorOfRef &lams,
                  const VectorOfRef &lams_ext,
                  std::vector<VectorRef> &lams_cache) const;

  /// @brief  Set the merit function penalty parameter.
  void setPenalty(const Scalar &new_mu) noexcept;

protected:
  /// AL penalty parameter
  Scalar mu_penal_;
  /// Reciprocal penalty parameter
  Scalar mu_inv_ = 1. / mu_penal_;
};

} // namespace proxnlp

#include "proxnlp/pdal.hxx"
