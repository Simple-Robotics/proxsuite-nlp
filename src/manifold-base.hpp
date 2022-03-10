#pragma once

#include <Eigen/Core>

#include "lienlp/fwd.hpp"


namespace lienlp {

  /// Macro which brings manifold typedefs up into the constraint, cost type, etc.
  #define LIENLP_DEFINE_INTERFACE_TYPES(M)    \
    using Scalar = typename M::Scalar;        \
    using Point_t = typename M::Point_t;      \
    using Vec_t = typename M::TangentVec_t;   \
    using Hess_t = Eigen::Matrix<Scalar, M::NV, M::NV, M::Options>;

  /**
   * Base class for manifolds, to use in cost funcs, solvers...
   */
  template<class T>
  struct ManifoldTpl {
  protected:
    using Self = ManifoldTpl<T>; /// Shorthand for the type of `this`

  public:
    using Scalar = typename traits<T>::Scalar; /// Scalar type
    enum {
      NQ = traits<T>::NQ,
      NV = traits<T>::NV,
      Options = traits<T>::Options
    };

    using Point_t = Eigen::Matrix<Scalar, NQ, 1, Options>;
    using TangentVec_t = Eigen::Matrix<Scalar, NV, 1, Options>;
    using Jac_t = Eigen::Matrix<Scalar, NV, NV, Options>; 

    T& derived()
    {
      return static_cast<T&>(*this);
    };

    const T& derived() const
    {
      return static_cast<const T&>(*this);
    }

    /// @brief    Get manifold representation dimension.
    int nx() const;
    /// @brief    Get manifold tangent space dimension.
    int ndx() const;

    /// @brief    Get the neutral element from the manifold (if this makes sense).
    Point_t zero() const { return derived().zero_impl(); }
    /// @brief    Sample a random point on the manifold.
    Point_t rand() const { return derived().rand_impl(); }

    /// @name     Operations

    /**
     * Perform the manifold integration operation.
     * This is an interface. Specific implementations should be in the derived classes.
     * 
     */
    template<class Vec_t, class Tangent_t>
    void integrate(const Eigen::MatrixBase<Vec_t>& x,
                   const Eigen::MatrixBase<Tangent_t>& v,
                   Eigen::MatrixBase<Vec_t>& out) const;

    /**
     * @brief   Jacobian of the integation operation.
     */
    template<int arg, class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate(const Eigen::MatrixBase<Vec_t>& x,
                    const Eigen::MatrixBase<Tangent_t>& v,
                    Eigen::MatrixBase<Jout_t>& Jout) const
    {
      derived().template Jintegrate_impl<arg>(x.derived(), v.derived(), Jout.derived());
    }

    /// @copybrief  Jintegrate()
    template<class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate(const Eigen::MatrixBase<Vec_t>& x,
                    const Eigen::MatrixBase<Tangent_t>& v,
                    Eigen::MatrixBase<Jout_t>& Jout,
                    int arg) const;

    /**
     * Perform the manifold retraction operation.
     */
    template<class Vec_t, class Tangent_t>
    void difference(const Eigen::MatrixBase<Vec_t>& x0,
              const Eigen::MatrixBase<Vec_t>& x1,
              Eigen::MatrixBase<Tangent_t>& out) const;

    template<int arg, class Vec_t, class Jout_t>
    void Jdifference(const Eigen::MatrixBase<Vec_t>& x0,
               const Eigen::MatrixBase<Vec_t>& x1,
               Eigen::MatrixBase<Jout_t>& Jout) const
    {
      derived().template Jdifference_impl<arg>(x0.derived(), x1.derived(), Jout.derived());
    }

    template<class Vec_t, class Jout_t>
    void Jdifference(const Eigen::MatrixBase<Vec_t>& x0,
               const Eigen::MatrixBase<Vec_t>& x1,
               Eigen::MatrixBase<Jout_t>& Jout,
               int arg) const;

    /// \name Default impls
    /// \{

    /// @copybrief integrate()
    ///
    /// Out-of-place variant.
    template<class Vec_t, class Tangent_t>
    Point_t integrate(const Eigen::MatrixBase<Vec_t>& x,
                      const Eigen::MatrixBase<Tangent_t>& v) const;

    /// @copybrief difference()
    ///
    /// Out-of-place version of diff operator.
    template<class Vec_t>
    TangentVec_t difference(const Eigen::MatrixBase<Vec_t>& x0,
                      const Eigen::MatrixBase<Vec_t>& x1) const;

    /// \}

  };

}  // namespace lienlp

#include "lienlp/manifold-base.hxx"
