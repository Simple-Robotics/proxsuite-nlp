#pragma once


#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {
  
  #define LIENLP_RESIDUAL_TYPES(Scalar)                                           \
    using ReturnType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;                     \
    using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>; \

  /**
   * @brief   Base template for constraint/residual functors.
   * 
   * Base template for constraint/residual functors. These should be
   * passed around to constraint classes (e.g. equality constraints) or cost
   * functions such as quadratic penalties.
   */
  template<typename _Scalar>
  struct ResidualBase
  {
  protected:
    const int m_nx;
    const int m_ndx;
    const int m_nr;
  public:
    using Scalar = _Scalar;
    LIENLP_RESIDUAL_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    /// @brief      Evaluate the residual at a given point x.
    virtual ReturnType operator()(const ConstVectorRef& x) const = 0;
    /// @brief      Jacobian matrix of the constraint function.
    virtual void computeJacobian(const ConstVectorRef& x, Eigen::Ref<JacobianType> Jout) const = 0;

    ResidualBase(const int nx, const int ndx, const int nr)
    : m_nx(nx), m_ndx(ndx), m_nr(nr) {}

    virtual ~ResidualBase<Scalar>() = default;

    int nx() const { return m_nx; }
    int ndx() const { return m_ndx; }
    int nr() const { return m_nr; }

    /** @copybrief computeJacobian()
     * 
     * Allocated version of the computeJacobian() method.
     */
    JacobianType computeJacobian(const ConstVectorRef& x) const
    {
      JacobianType Jout(m_nr, m_ndx);
      computeJacobian(x, Jout);
      return Jout;
    }

    /// Vector-hessian product.
    virtual void vhp(const ConstVectorRef& x, const ConstVectorRef& v, RefMatrix Hout) const
    {
      Hout.setZero();
    }

    JacobianType vhp(const ConstVectorRef& x, const ConstVectorRef& v) const
    {
      JacobianType J(m_ndx, m_ndx);
      vhp(x, v, J);
      return J;
    }

  };

} // namespace lienlp
