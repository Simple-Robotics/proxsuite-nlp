/** @file Base definitions for functor classes.
 */
#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"

namespace lienlp
{
  #define LIENLP_FUNCTOR_TYPEDEFS(Scalar)                                       \
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)                                         \
    using ReturnType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;                \
    using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>; \

  /**
   * @brief Base functor type.
   */
  template<typename _Scalar>
  struct BaseFunctor : math_types<_Scalar>
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

    BaseFunctor(const int nx, const int ndx, const int nr)
      : m_nx(nx), m_ndx(ndx), m_nr(nr) {}

    virtual ~BaseFunctor() = default;

    int nx() const { return m_nx; }
    int ndx() const { return m_ndx; }
    int nr() const { return m_nr; }
  };

  /** @brief  Differentiable functor, with methods to compute both Jacobians and vector-hessian products.
   */
  template<typename _Scalar>
  struct DifferentiableFunctor : BaseFunctor<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    LIENLP_FUNCTOR_TYPEDEFS(Scalar)

    DifferentiableFunctor(const int nx, const int ndx, const int nr)
      : BaseFunctor<Scalar>(nx, ndx, nr) {}

    /// @brief      Jacobian matrix of the constraint function.
    virtual void computeJacobian(const ConstVectorRef& x, MatrixRef Jout) const = 0;
    /// @brief      Vector-hessian product.
    virtual void vectorHessianProduct(const ConstVectorRef&, const ConstVectorRef&, MatrixRef Hout) const
    {
      Hout.setZero();
    }
  };

} // namespace lienlp

