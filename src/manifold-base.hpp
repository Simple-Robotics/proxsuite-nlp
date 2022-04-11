#pragma once

#include <Eigen/Core>

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"


namespace lienlp
{

  /// Macro which brings manifold typedefs up into the constraint, cost type, etc.
  #define LIENLP_DEFINE_MANIFOLD_TYPES(M)                     \
    LIENLP_DYNAMIC_TYPEDEFS(typename M::Scalar)           \
    using PointType = typename M::PointType;                  \
    using TangentVectorType = typename M::TangentVectorType;  \
    using JacobianType = typename M::JacobianType;

  /**
   * Base class for manifolds, to use in cost funcs, solvers...
   */
  template<typename _Scalar, int _Options>
  struct ManifoldAbstractTpl {
  public:
    using Scalar = _Scalar; /// Scalar type
    enum {
      Options = _Options
    };

    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using PointType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
    using TangentVectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
    using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>; 

    virtual ~ManifoldAbstractTpl() = default;

    /// @brief    Get manifold representation dimension.
    virtual int nx() const = 0;
    /// @brief    Get manifold tangent space dimension.
    virtual int ndx() const = 0;

    /// @brief    Get the neutral element \f$e \in M\f$ from the manifold (if this makes sense).
    virtual PointType neutral() const { return PointType::Zero(nx()); }
    /// @brief    Sample a random point \f$x \in M\f$ on the manifold.
    virtual PointType rand() const { return PointType::Zero(nx()); }

    /// @name     Operations

    /** Perform the manifold integration operation.
     * 
     * @details This is an interface. Specific implementations should be in the derived classes.
     */
    virtual void integrate_impl(const ConstVectorRef& x,
                                const ConstVectorRef& v,
                                VectorRef out) const = 0;

    void integrate(const ConstVectorRef& x,
                   const ConstVectorRef& v,
                   VectorRef out) const
    {
      integrate_impl(x, v, out);
    }

    /** @brief   Jacobian of the integation operation.
     */
    virtual void Jintegrate_impl(const ConstVectorRef& x,
                                 const ConstVectorRef& v,
                                 MatrixRef Jout,
                                 int arg) const = 0;

    void Jintegrate(const ConstVectorRef& x,
                    const ConstVectorRef& v,
                    MatrixRef Jout,
                    int arg) const
    {
      Jintegrate_impl(x, v, Jout, arg);
    }

    /// @brief Perform the manifold retraction operation.
    virtual void difference_impl(const ConstVectorRef& x0,
                                 const ConstVectorRef& x1,
                                 VectorRef out) const = 0;

    void difference(const ConstVectorRef& x0,
                    const ConstVectorRef& x1,
                    VectorRef out) const
    {
      difference_impl(x0, x1, out);
    }

    /// @brief    Jacobian of the retraction operation.
    virtual void Jdifference_impl(const ConstVectorRef& x0,
                                  const ConstVectorRef& x1,
                                  MatrixRef Jout,
                                  int arg) const = 0;

    void Jdifference(const ConstVectorRef& x,
                     const ConstVectorRef& v,
                     MatrixRef Jout,
                     int arg) const
    {
      Jdifference_impl(x, v, Jout, arg);
    }

    /// @brief    Interpolation operation.
    virtual void interpolate_impl(const ConstVectorRef& x0,
                                  const ConstVectorRef& x1,
                                  const Scalar& u,
                                  VectorRef out) const = 0;

    void interpolate(const ConstVectorRef& x0,
                     const ConstVectorRef& x1,
                     const Scalar& u,
                     VectorRef out) const
    {
      interpolate_impl(x0, x1, u, out);
    }

    /// \name Out-of-place (allocated) overloads.
    /// \{

    /// @copybrief integrate()
    ///
    /// Out-of-place variant of integration operator.
    PointType integrate(const ConstVectorRef& x,
                        const ConstVectorRef& v) const
    {
      PointType out(nx());
      integrate_impl(x, v, out);
      return out;
    }

    /// @copybrief difference()
    ///
    /// Out-of-place version of diff operator.
    TangentVectorType difference(const ConstVectorRef& x0,
                                 const ConstVectorRef& x1) const
    {
      TangentVectorType out(ndx());
      difference_impl(x0, x1, out);
      return out;
    }

    /// @copybrief interpolate_impl()
    PointType interpolate(const ConstVectorRef& x0,
                          const ConstVectorRef& x1,
                          const Scalar& u) const
    {
      PointType out(nx());
      interpolate_impl(x0, x1, u, out);
      return out;
    }

    /// \}

  };

}  // namespace lienlp
