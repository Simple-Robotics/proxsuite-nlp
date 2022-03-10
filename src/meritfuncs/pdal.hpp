#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/merit-function-base.hpp"

#include <iostream>
#include <vector>


namespace lienlp {
  
  /**
   * Primal-dual Augmented Lagrangian function, extending
   * the function from Gill & Robinson (2012) to inequality constraints.
   * 
   */
  template<typename _Scalar>
  struct PDALFunction :
    MeritFunctionTpl<_Scalar,
      typename math_types<_Scalar>::VectorList,
      typename math_types<_Scalar>::VectorList>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Prob_t = Problem<Scalar>;

    Prob_t* m_prob;

    /// AL penalty parameter
    Scalar m_muEq = Scalar(0.01);
    /// Generalized pdAL dual penalty param
    const Scalar m_gamma = Scalar(1.);

    /// Set the merit function penalty parameter.
    void setPenalty(const Scalar& new_mu)
    {
      m_muEq = new_mu;
    };

    PDALFunction(Prob_t* prob) : m_prob(prob) {}

    Scalar operator()(const VectorXs& x, const VectorList& lams, const VectorList& lams_ext) const
    {
      Scalar result_ = Scalar(0.);
      result_ = result_ + m_prob->m_cost(x);
      std::vector<VectorXs> displaced_residuals_;
      const std::size_t num_c = m_prob->getNumConstraints();

      for (std::size_t i = 0; i < num_c; i++)
      {
        auto eq_cstr = m_prob->getEqCs(i);

        // constraint value
        VectorXs cval = eq_cstr->operator()(x);
        // displace the constraint
        cval.noalias() += m_muEq * lams_ext[i];
        // projection
        cval.noalias() = eq_cstr->projection(cval);
        displaced_residuals_.push_back(cval);

        result_ += (Scalar(0.5) / m_muEq) * cval.dot(cval);

        VectorXs dual_res = cval - m_muEq * lams[i];
        result_ += (Scalar(0.5) * m_gamma / m_muEq) * dual_res.dot(dual_res);
      }

      return result_;
    }

    VectorXs gradient(const VectorXs& x, const VectorList& lams, const VectorList& lams_ext) const
    {
      VectorXs g = m_prob->m_cost.gradient(x);
      return g;
    }

  };

}