#pragma once

#include "lienlp/macros.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp {

  #define LIENLP_CSTR_TYPES(Scalar)                       \
    using C_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>; \
    using Jacobian_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

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
    using Scalar = _Scalar;
    LIENLP_CSTR_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    virtual C_t operator()(const VectorXs& x) const = 0;
    /// @brief      Jacobian matrix of the constraint function.
    virtual void jacobian(const VectorXs& x, Jacobian_t& Jout) const = 0;

    /** @copybrief jacobian()
     * 
     * Allocated version of the jacobian() method.
     */
    Jacobian_t jacobian(const VectorXs& x) const
    {
      Jacobian_t Jout;
      jacobian(x, Jout);
      return Jout;
    }

    /// Vector-hessian product.
    virtual Jacobian_t vhp(const VectorXs& x, const VectorXs& v) const
    {
      Jacobian_t J;
      J.resize(1, 1);
      J.setZero();
      return J;
    }

    virtual ~ConstraintFuncTpl<Scalar>() = default;

  };


  /**
   * @brief   Constraint format: negative/positive orthant, cones, etc...
   */
  template<typename _Scalar>
  struct ConstraintFormatBaseTpl
  {
  protected:
    const int m_nc;
  public:
    using Scalar = _Scalar;
    LIENLP_CSTR_TYPES(Scalar)
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    using functor_t = ConstraintFuncTpl<Scalar>;
    const functor_t& m_func;

    ConstraintFormatBaseTpl<Scalar>(const functor_t& func, const int& nc)
      : m_func(func), m_nc(nc) {}

    inline typename functor_t::C_t operator()(const VectorXs& x) const
    {
      return m_func(x);
    }

    inline typename functor_t::Jacobian_t jacobian(const VectorXs& x) const
    {
      return m_func.jacobian(x);
    }

    /// Get dimension of constraint representation.
    int getDim() const
    {
      return m_nc;
    }

    virtual C_t projection(const VectorXs& z) const = 0;
    inline C_t dualProjection(const VectorXs& z) const
    {
      return z - projection(z);
    }
    virtual Jacobian_t Jprojection(const VectorXs& z) const = 0;

    virtual ~ConstraintFormatBaseTpl<Scalar>() = default;
  };

}

#include "lienlp/constraint-base.hxx"
