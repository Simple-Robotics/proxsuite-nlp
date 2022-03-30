#pragma once


#include "lienlp/functor-base.hpp"
#include "lienlp/functor-ops.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"


namespace lienlp
{
  
  /**
   * @brief Linear residuals \f$r(x) = Ax + b\f$.
   */
  template<typename _Scalar>
  struct LinearResidual : DifferentiableFunctor<_Scalar>
  {
    using Scalar = _Scalar;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    using Base = DifferentiableFunctor<Scalar>;
    using Base::computeJacobian;

    const MatrixXs mat;
    const VectorXs b;

    LinearResidual(const ConstMatrixRef& A, const ConstVectorRef& b)
      : Base((int)A.cols(), (int)A.cols(), (int)A.rows()),
        mat(A),
        b(b) {}

    ReturnType operator()(const ConstVectorRef& x) const
    {
      return mat * x + b;
    }

    void computeJacobian(const ConstVectorRef&, Eigen::Ref<JacobianType> Jout) const
    {
      Jout = mat;
    }
  };


  /** @brief    Linear function of difference vector on a manifold, of the form
   *            \f$ r(x) = A(x \ominus \bar{x}) + b \f$.
   */
  template<typename _Scalar>
  struct LinearStateResidual : ComposeFunctor<_Scalar>
  {
    using Scalar = _Scalar;
    using Base = ComposeFunctor<Scalar>;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    using M = ManifoldAbstract<Scalar>;

    LinearStateResidual(const M& manifold, const ConstVectorRef& target,
                        const ConstMatrixRef& A, const ConstVectorRef& b)
                        : Base(LinearResidual<Scalar>(A, b),
                               StateResidual<Scalar>(manifold, target)
                               ) {}

  };



} // namespace lienlp
