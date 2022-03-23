#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp {

  /**
   * @brief   Negative orthant, for constraints \f$h(x)\leq 0\f$.
   * 
   * Negative orthant, corresponding to constraints of the form
   * \f[
   *    h(x) \leq 0
   * \f]
   * where \f$h : \mathcal{X} \to \RR^p\f$ is a given residual.
   */
  template<typename _Scalar>
  struct NegativeOrthant : ConstraintSetBase<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Base = ConstraintSetBase<Scalar>;
    using Base::operator();
    using Active_t = typename Base::Active_t;
    using functor_t = typename Base::functor_t;

    NegativeOrthant(const functor_t& func) : Base(func) {}

    ReturnType projection(const ConstVectorRef& z) const
    {
      return z.cwiseMin(Scalar(0.));
    }

    JacobianType Jprojection(const ConstVectorRef& z) const
    {
      const int nr = this->nr();
      Active_t active_set(nr);
      computeActiveSet(z, active_set);
      JacobianType Jout(nr, nr);
      Jout.setIdentity();
      for (int i = 0; i < nr; i++)
      {
        if (active_set(i))
        {
          Jout.col(i).setZero();
        }
      }
      return Jout;
    }

    void computeActiveSet(const ConstVectorRef& z,
                          Eigen::Ref<Active_t> out) const
    {
      out.array() = (z.array() > Scalar(0.));
    }

  };

} // namespace lienlp
