#pragma once


#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"
#include "lienlp/functor-base.hpp"


namespace lienlp
{

  /**
   * @brief   Base template for constraint/residual functors.
   * 
   * Base template for constraint/residual functors. These should be
   * passed around to constraint classes (e.g. equality constraints) or cost
   * functions such as quadratic penalties.
   */
  template<typename _Scalar>
  struct ResidualBase : DifferentiableFunctor<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    using Base = DifferentiableFunctor<Scalar>;
    using Base::computeJacobian;
    using Base::vectorHessianProduct;

    ResidualBase(const int nx, const int ndx, const int nr)
    : Base(nx, ndx, nr) {}

    /** @copybrief computeJacobian()
     * 
     * Allocated version of the computeJacobian() method.
     */
    JacobianType computeJacobian(const ConstVectorRef& x) const
    {
      JacobianType Jout(this->nr(), this->ndx());
      computeJacobian(x, Jout);
      return Jout;
    }

    JacobianType vectorHessianProduct(const ConstVectorRef& x, const ConstVectorRef& v) const
    {
      JacobianType J(this->ndx(), this->ndx());
      vectorHessianProduct(x, v, J);
      return J;
    }

  };

} // namespace lienlp
