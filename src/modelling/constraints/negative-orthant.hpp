#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/constraint-base.hpp"


namespace lienlp
{

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
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    using Base = ConstraintSetBase<Scalar>;
    using ActiveType = typename Base::ActiveType;
    using FunctionType = typename Base::FunctionType;

    explicit NegativeOrthant(const FunctionType& func) : Base(func) {}

    ReturnType projection(const ConstVectorRef& z) const
    {
      return z.cwiseMin(Scalar(0.));
    }

    /// The elements of the active set are the components such that \f$z_i > 0\f$.
    void computeActiveSet(const ConstVectorRef& z,
                          Eigen::Ref<ActiveType> out) const
    {
      out.array() = (z.array() > Scalar(0.));
    }

  };

} // namespace lienlp
