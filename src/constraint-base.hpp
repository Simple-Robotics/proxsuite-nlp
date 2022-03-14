#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {

  #define LIENLP_CSTR_TYPES(Scalar)                                           \
    using C_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;                     \
    using Jacobian_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>; \
    using Active_t = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

  /**
   * @brief   Base template for constraint/residual functors.
   * 
   * Base template for constraint/residual functors. These should be
   * passed around to constraint classes (e.g. equality constraints) or cost
   * functions such as quadratic penalties.
   */
  template<typename _Scalar>
  struct ConstraintFuncTpl
  {
  protected:
    const int m_nc;
    const int m_ndx;
  public:
    using Scalar = _Scalar;
    LIENLP_CSTR_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    virtual C_t operator()(const ConstVectorRef& x) const = 0;
    /// @brief      Jacobian matrix of the constraint function.
    virtual void jacobian(const ConstVectorRef& x, Jacobian_t& Jout) const = 0;

    ConstraintFuncTpl(const int& nc, const int& ndx)
    : m_nc(nc), m_ndx(ndx) {}

    virtual ~ConstraintFuncTpl<Scalar>() = default;

    int getDim() const { return m_nc; }
    int ndx() const { return m_ndx; }

    /** @copybrief jacobian()
     * 
     * Allocated version of the jacobian() method.
     */
    Jacobian_t jacobian(const ConstVectorRef& x) const
    {
      Jacobian_t Jout(m_nc, m_ndx);
      jacobian(x, Jout);
      return Jout;
    }

    /// Vector-hessian product.
    virtual Jacobian_t vhp(const ConstVectorRef& x, const ConstVectorRef& v) const
    {
      Jacobian_t J(m_ndx, m_ndx);
      J.setZero();
      return J;
    }

  };


  /**
   * @brief   Constraint format: negative/positive orthant, cones, etc...
   */
  template<typename _Scalar>
  struct ConstraintFormatBaseTpl
  {
  public:
    using Scalar = _Scalar;
    LIENLP_CSTR_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    using functor_t = ConstraintFuncTpl<Scalar>;
    const functor_t& m_func;

    ConstraintFormatBaseTpl<Scalar>(const functor_t& func)
      : m_func(func) {}

    inline C_t operator()(const ConstVectorRef& x) const
    {
      return m_func(x);
    }

    inline Jacobian_t jacobian(const ConstVectorRef& x) const
    {
      return m_func.jacobian(x);
    }

    /// Get dimension of constraint representation.
    int getDim() const { return m_func.getDim(); }
    /// Get tangent space dimension (no. of columns of Jacobian)
    int ndx() const { return m_func.ndx(); }

    virtual C_t projection(const ConstVectorRef& z) const = 0;
    inline C_t dualProjection(const ConstVectorRef& z) const
    {
      return z - projection(z);
    }
    /// Compute the jacobian of the active-set projection operator.
    virtual Jacobian_t Jprojection(const ConstVectorRef& z) const = 0;
    /// Compute the active set of the constraint.
    virtual void computeActiveSet(const ConstVectorRef& z, Active_t& out) const = 0;
    virtual ~ConstraintFormatBaseTpl<Scalar>() = default;
  };

}

#include "lienlp/constraint-base.hxx"
