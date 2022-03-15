#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp {

  template<typename _Scalar>
  struct NegativeOrthant : ConstraintSetBase<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar);
    using Base = ConstraintSetBase<Scalar>;
    using Base::operator();
    using Base::computeJacobian;
    using Base::nc;
    using Base::ndx;
    using Base::Active_t;

    ReturnType projection(const ConstVectorRef& z) const
    {
      return z.cwiseMin(Scalar(0.));
    }

    JacobianType Jprojection(const ConstVectorRef& z) const
    {
      Active_t active_set(nr());
      computeActiveSet(z, active_set);
      JacobianType Jout(nr(), ndx());
      Jout.setIdentity();
      for (const int i = 0; i < nr(); i++)
      {
        Jout.row(i).setZero();
      }
      return Jout;
    }

    void computeActiveSet(const ConstVectorRef& z,
                          Active_t& out) const
    {
      out.array() = (z >= Scalar(0.));
    }


  }

} // namespace lienlp
