#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp
{

  /**
   * @brief   Composite \f$\ell_1\f$-penalty function.
   * 
   * @details The composite \f$\ell_1\f$-penalty penalizes the norm
   *          \f$ \| r(x) \|_1\f$ of a residual function.
   *          This class implements the proximity operator (soft-thresholding)
   *          and an appropriate generalized Jacobian.
   */
  template <typename _Scalar>
  struct L1Penalty : ConstraintSetBase<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    using Base = ConstraintSetBase<Scalar>;
    using Base::operator();
    using Active_t = typename Base::Active_t;
    using functor_t = typename Base::functor_t;

    Scalar m_mu = 0.01;

    L1Penalty(const functor_t& func) : Base(func) {}

    ReturnType projection(const ConstVectorRef& z) const
    {
      return z.array().sign() * (z.abs().array() - m_mu).max(Scalar(0.));
    }

    void computeActiveSet(const ConstVectorRef& z,
                          Eigen::Ref<Active_t> out) const
    {
      out = (z.abs().array() - m_mu) <= Scalar(0.);
    }

    void updateProxParameters(const Scalar mu) {
      m_mu = mu;
    };

  };

}
