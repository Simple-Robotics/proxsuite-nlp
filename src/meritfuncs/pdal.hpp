#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/merit-function-base.hpp"
#include "lienlp/meritfuncs/lagrangian.hpp"

#include <vector>
#include <boost/shared_ptr.hpp>

namespace lienlp {
  using boost::shared_ptr;

  /**
   * Primal-dual Augmented Lagrangian function, extending
   * the function from Gill & Robinson (2012) to inequality constraints.
   * 
   */
  template<typename _Scalar>
  struct PDALFunction :
    public MeritFunctorBase<
      _Scalar,
      typename math_types<_Scalar>::VectorList,
      typename math_types<_Scalar>::VectorList>
  {
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Parent = MeritFunctorBase<Scalar, VectorList, VectorList>;
    using Parent::m_prob;
    using Parent::gradient;
    using Prob_t = Problem<Scalar>;
    using Lagrangian_t = LagrangianFunction<Scalar>;

    Lagrangian_t m_lagr;

    /// AL penalty parameter
    Scalar m_muEq = Scalar(0.01);

    /// Generalized pdAL dual penalty param
    const Scalar m_gamma = Scalar(1.);

    /// Set the merit function penalty parameter.
    void setPenalty(const Scalar& new_mu) { m_muEq = new_mu; };

    /// Get the merit function penalty parameter;
    const Scalar& getPenalty() { return m_muEq; }

    PDALFunction(shared_ptr<Prob_t> prob)
      : Parent(prob), m_lagr(Lagrangian_t(prob)) {}

    /**
     *  @brief Compute the first-order multiplier estimates.
     */
    void computeFirstOrderMultipliers(
      const VectorXs& x,
      const VectorList& lams_ext,
      VectorList& out) const
    {
      for (std::size_t i = 0; i < m_prob->getNumConstraints(); i++)
      {
        auto cstr = m_prob->getCstr(i);
        out.push_back((*cstr)(x));  // constraint val
        out[i].noalias() = out[i] + lams_ext[i] / m_muEq;
        out[i].noalias() = cstr->dualProjection(out[i]);
      }
    }

    /// @brief Compute the pdAL (Gill-Robinson) multipliers
    void computePDALMultipliers(
      const VectorXs& x,
      const VectorList& lams,
      const VectorList& lams_ext,
      VectorList& out) const
    {
      computeFirstOrderMultipliers(x, lams_ext, out);
      for (std::size_t i = 0; i < m_prob->getNumConstraints(); i++)
      {
        out[i].noalias() = 2 * out[i] - lams[i] / m_muEq;
      }
    }

    /// @copybrief computeFirstOrderMultipliers()
    /// Out-of-place variant.
    VectorList computeFirstOrderMultipliers(
      const VectorXs& x,
      const VectorList& lams_ext) const
    {
      VectorList out;
      const std::size_t num_c = m_prob->getNumConstraints();
      out.reserve(num_c);
      computeFirstOrderMultipliers(x, lams_ext, out);
      return out;
    }

    Scalar operator()(const VectorXs& x, const VectorList& lams, const VectorList& lams_ext) const
    {
      Scalar result_ = Scalar(0.);
      result_ = result_ + m_prob->m_cost(x);
      VectorList displaced_residuals_ = computeFirstOrderMultipliers(x, lams_ext);
      const std::size_t num_c = m_prob->getNumConstraints();

      for (std::size_t i = 0; i < num_c; i++)
      {
        VectorXs cval = displaced_residuals_[i];
        result_ += (Scalar(0.5) / m_muEq) * cval.dot(cval);
        // dual penalty
        VectorXs dual_res = cval - m_muEq * lams[i];
        result_ += (Scalar(0.5) * m_gamma / m_muEq) * dual_res.dot(dual_res);
      }

      return result_;
    }

    void gradient(const VectorXs& x,
                  const VectorList& lams,
                  const VectorList& lams_ext,
                  VectorXs& out) const;

    void hessian(const VectorXs& x,
                 const VectorList& lams,
                 const VectorList& lams_ext,
                 MatrixXs& out) const;
  };

}

#include "lienlp/meritfuncs/pdal.hxx"