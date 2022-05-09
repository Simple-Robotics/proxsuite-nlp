#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/merit-function-base.hpp"
#include "proxnlp/lagrangian.hpp"

#include <vector>

namespace proxnlp
{

  /**
   * @brief   Primal-dual augmented Lagrangian-type merit function.
   * 
   * Primal-dual Augmented Lagrangian function, extending
   * the function from Gill & Robinson (2012) to inequality constraints.
   * For inequality constraints of the form \f$ c(x) \in \calC \f$ and an objective function
   * \f$ f\colon\calX \to \RR \f$,
   * \f[
   *    \calM_{\mu}(x, \lambda; \lambda_e) = f(x) + \frac{1}{2\mu} \| \proj_\calC(c(x) + \mu \lambda_e) \|_2^2
   *    + \frac{1}{2\mu} \| \proj_\calC(c(x) + \mu\lambda_e) - \mu\lambda) \|_2^2.
   * \f]
   * 
   */
  template<typename _Scalar>
  struct PDALFunction :
    public MeritFunctionBase<
      _Scalar,
      typename math_types<_Scalar>::VectorOfRef,
      typename math_types<_Scalar>::VectorOfRef>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Base = MeritFunctionBase<Scalar, VectorOfRef, VectorOfRef>;
    using Base::m_prob;
    using Problem = ProblemTpl<Scalar>;
    using Lagrangian_t = LagrangianFunction<Scalar>;

    Lagrangian_t m_lagr;

    /// AL penalty parameter
    Scalar m_mu;
    /// Reciprocal penalty parameter
    Scalar m_muInv = 1. / m_mu;

    /// Generalized pdAL dual penalty param
    const Scalar m_gamma = 1.;

    /// Set the merit function penalty parameter.
    void setPenalty(const Scalar& new_mu)
    {
      m_mu = new_mu;
      m_muInv = 1. / new_mu;
    };

    PDALFunction(shared_ptr<Problem> prob, const Scalar mu = 0.01)
      : Base(prob)
      , m_lagr(Lagrangian_t(prob))
      , m_mu(mu)
      {}

    /**
     *  @brief Compute the first-order multiplier estimates.
     */
    void computeFirstOrderMultipliers(
      const ConstVectorRef& x,
      const VectorOfRef& lams_ext,
      VectorOfRef& out) const;

    /// @brief Compute the pdAL (Gill-Robinson) multipliers
    /// @todo  Use this at some point in the algorithm
    void computePDALMultipliers(
      const ConstVectorRef& x,
      const VectorOfRef& lams,
      const VectorOfRef& lams_ext,
      VectorOfRef& out) const;

    Scalar operator()(const ConstVectorRef& x,
                      const VectorOfRef& lams,
                      const VectorOfRef& lams_ext) const;

    void computeGradient(const ConstVectorRef& x,
                         const VectorOfRef& lams,
                         const VectorOfRef& lams_ext,
                         VectorRef out) const;
  };

}

#include "proxnlp/pdal.hxx"