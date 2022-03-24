#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/merit-function-base.hpp"
#include "lienlp/meritfuncs/lagrangian.hpp"

#include <vector>

namespace lienlp {

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
    public MeritFunctorBase<
      _Scalar,
      typename math_types<_Scalar>::VectorOfVectors,
      typename math_types<_Scalar>::VectorOfVectors>
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Base = MeritFunctorBase<Scalar, VectorOfVectors, VectorOfVectors>;
    using Base::m_prob;
    using Base::computeGradient;
    using Prob_t = Problem<Scalar>;
    using Lagrangian_t = LagrangianFunction<Scalar>;

    Lagrangian_t m_lagr;

    /// AL penalty parameter
    Scalar m_muEq = 0.01;

    /// Generalized pdAL dual penalty param
    const Scalar m_gamma = 1.;

    /// Set the merit function penalty parameter.
    void setPenalty(const Scalar& new_mu) { m_muEq = new_mu; };

    /// Get the merit function penalty parameter;
    const Scalar& getPenalty() { return m_muEq; }

    PDALFunction(shared_ptr<Prob_t> prob)
      : Base(prob), m_lagr(Lagrangian_t(prob)) {}

    /**
     *  @brief Compute the first-order multiplier estimates.
     */
    void computeFirstOrderMultipliers(
      const ConstVectorRef& x,
      const VectorOfVectors& lams_ext,
      VectorOfVectors& out) const
    {
      for (std::size_t i = 0; i < m_prob->getNumConstraints(); i++)
      {
        auto cstr = m_prob->getConstraint(i);
        out.push_back((*cstr)(x) + lams_ext[i] / m_muEq);
        out[i].noalias() = cstr->normalConeProjection(out[i]);
      }
    }

    /// @brief Compute the pdAL (Gill-Robinson) multipliers
    /// @todo   fix recomputing 1st order multipliers (w/ workspace)
    void computePDALMultipliers(
      const ConstVectorRef& x,
      const VectorOfVectors& lams,
      const VectorOfVectors& lams_ext,
      VectorOfVectors& out) const
    {
      // TODO fix calling this again; grab values from workspace
      computeFirstOrderMultipliers(x, lams_ext, out);
      for (std::size_t i = 0; i < m_prob->getNumConstraints(); i++)
      {
        out[i].noalias() = 2 * out[i] - lams[i] / m_muEq;
      }
    }

    Scalar operator()(const ConstVectorRef& x,
                      const VectorOfVectors& lams,
                      const VectorOfVectors& lams_ext) const
    {
      Scalar result_ = m_prob->m_cost(x);

      const std::size_t num_c = m_prob->getNumConstraints();
      for (std::size_t i = 0; i < num_c; i++)
      {
        auto cstr = m_prob->getConstraint(i);
        VectorXs cval = (*cstr)(x) + m_muEq * lams_ext[i];
        cval.noalias() = cstr->normalConeProjection(cval);
        result_ += (Scalar(0.5) / m_muEq) * cval.squaredNorm();
        // dual penalty
        VectorXs dual_res = cval - m_muEq * lams[i];
        result_ += (Scalar(0.5) / m_muEq) * dual_res.squaredNorm();
      }

      return result_;
    }

    void computeGradient(const ConstVectorRef& x,
                         const VectorOfVectors& lams,
                         const VectorOfVectors& lams_ext,
                         VectorRef out) const;

    void computeHessian(const ConstVectorRef& x,
                        const VectorOfVectors& lams,
                        const VectorOfVectors& lams_ext,
                        MatrixRef out) const;
  };

}

#include "lienlp/meritfuncs/pdal.hxx"