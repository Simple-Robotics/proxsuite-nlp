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
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    using Base = ConstraintSetBase<Scalar>;
    using ActiveType = typename Base::ActiveType;
    using FunctionType = typename Base::FunctionType;

    Scalar m_mu = 0.01;

    explicit L1Penalty(const FunctionType& func) : Base(func) {}

    ReturnType projection(const ConstVectorRef& z) const
    {
      return z.array().sign() * (z.abs().array() - m_mu).max(Scalar(0.));
    }

    void computeActiveSet(const ConstVectorRef& z,
                          Eigen::Ref<ActiveType> out) const
    {
      out = (z.abs().array() - m_mu) <= Scalar(0.);
    }

    void updateProxParameters(const Scalar mu) {
      m_mu = mu;
    };

  };

}
