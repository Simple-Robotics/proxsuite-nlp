#pragma once

#include <Eigen/Core>

#include "lienlp/fwd.hpp"


namespace lienlp {

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

    T& derived()
    {
      return static_cast<T&>(*this);
    };

    const T& derived() const
    {
      return static_cast<const T&>(*this);
    }

    int get_nq() const { return derived().nq_impl(); }  /// get repr dimension
    int get_nv() const { return derived().nv_impl(); }  /// get tangent space dim

    /**
     * Perform the manifold integration operation.
     * This is an interface. Specific implementations should be in the derived classes.
     * 
     */
    template<class Vec_t, class Tangent_t>
    void integrate(const Eigen::MatrixBase<Vec_t>& x,
                   const Eigen::MatrixBase<Tangent_t>& v,
                   Eigen::MatrixBase<Vec_t>& out) const
    {
      derived().integrate_impl(x.derived(), v.derived(), out.derived());
    }

    /**
     * Implementation of the integration operation.
     */
    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        Eigen::MatrixBase<Vec_t>& out) const;

    /**
     * Perform the manifold retraction operation.
     */
    template<class Vec_t, class Tangent_t>
    void diff(const Eigen::MatrixBase<Vec_t>& x0,
              const Eigen::MatrixBase<Vec_t>& x1,
              Eigen::MatrixBase<Tangent_t>& out) const
    {
      derived().diff_impl(x0.derived(), x1.derived(), out.derived());
    }

    template<class Vec_t, class Tangent_t>
    void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                   const Eigen::MatrixBase<Vec_t>& x1,
                   Eigen::MatrixBase<Tangent_t>& out) const;

    /// Out-of-place variant.
    template<class Vec_t, class Tangent_t>
    Point_t integrate(const Eigen::MatrixBase<Vec_t>& x,
                          const Eigen::MatrixBase<Tangent_t>& v) const
    {
      Point_t out;
      out.resize(get_nq());
      integrate(x, v, out);
      return out;
    }

  };

}

