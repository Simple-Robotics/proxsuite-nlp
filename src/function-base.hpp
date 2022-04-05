/** @file Base definitions for function classes.
 */
#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"

namespace lienlp
{
  /**
   * @brief Base function type.
   */
  template<typename _Scalar>
  struct BaseFunction : math_types<_Scalar>
  {
  protected:
    const int m_nx;
    const int m_ndx;
    const int m_nr;
  public:
    using Scalar = _Scalar;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    /// @brief      Evaluate the residual at a given point x.
    virtual ReturnType operator()(const ConstVectorRef& x) const = 0;

    BaseFunction(const int nx, const int ndx, const int nr)
      : m_nx(nx), m_ndx(ndx), m_nr(nr) {}

    virtual ~BaseFunction() = default;

    /// Get function input vector size (representation of manifold).
    int nx() const { return m_nx; }
    /// Get input manifold's tangent space dimension.
    int ndx() const { return m_ndx; }
    /// Get function codimension.
    int nr() const { return m_nr; }
  };

  /** @brief  Differentiable function, with method for the Jacobian.
   */
  template<typename _Scalar>
  struct C1Function : public BaseFunction<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    using Base = BaseFunction<_Scalar>;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    Base& toBase()
    {
      return static_cast<Base&>(*this);
    }

    C1Function(const int nx, const int ndx, const int nr)
      : Base(nx, ndx, nr) {}
  
    /// @brief      Jacobian matrix of the constraint function.
    virtual void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const = 0;

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

  };

  /** @brief  Twice-differentiable function, with methods to compute both Jacobians and vector-hessian products.
   */
  template<typename _Scalar>
  struct C2Function : public C1Function<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    using Base = C1Function<_Scalar>;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    Base& toC1()
    {
      return static_cast<Base&>(*this);
    }

    C2Function(const int nx, const int ndx, const int nr)
      : Base(nx, ndx, nr) {}

    /// @brief      Vector-hessian product.
    virtual void vectorHessianProduct(const ConstVectorRef&, const ConstVectorRef&, Eigen::Ref<JacobianType> Hout) const
    {
      Hout.setZero();
    }

    JacobianType vectorHessianProduct(const ConstVectorRef& x, const ConstVectorRef& v) const
    {
      JacobianType J(this->ndx(), this->ndx());
      vectorHessianProduct(x, v, J);
      return J;
    }

  };

} // namespace lienlp

