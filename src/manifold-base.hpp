#pragma once

#include <Eigen/Core>


namespace lienlp {

  template<class T>
  struct ManifoldTpl {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    T& derived()
    {
      return static_cast<T&>(*this);
    };

    const T& derived() const
    {
      return static_cast<const T&>(*this);
    }

    template<class Vec_t, class Tangent_t>
    void integrate(const Eigen::MatrixBase<Vec_t>& x,
                   const Eigen::MatrixBase<Tangent_t>& v,
                   Eigen::MatrixBase<Vec_t>& out) const
    {
      derived().integrate_impl(x.derived(), v.derived(), out.derived());
    }

    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        Eigen::MatrixBase<Vec_t>& out) const;

    template<class Vec_t, class Tangent_t>
    void diff(const Eigen::MatrixBase<Vec_t>& x0,
              const Eigen::MatrixBase<Vec_t>& x1,
              Eigen::MatrixBase<Tangent_t>& out) const
    {
      derived().diff_impl(x0.derived(), x1.derived(), out.derived());
    }

    // template<class Vec_t>
    // decltype(auto) diff(const Eigen::MatrixBase<Vec_t>& x0,
    //                     const Eigen::MatrixBase<Vec_t>& x1) const
    // {
      
    //   return diff()
    // }

    template<class Vec_t, class Tangent_t>
    void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                   const Eigen::MatrixBase<Vec_t>& x1,
                   Eigen::MatrixBase<Tangent_t>& out) const;

  };

  /// N-dimensional vector space.
  template<int Dim, typename Scalar, int _Options=0>
  struct VectorSpace : public ManifoldTpl<VectorSpace<Dim, Scalar, _Options>> {
    using scalar = Scalar;
    enum {
      NQ = Dim,
      Options = _Options
    };
    using Vec_t = Eigen::Matrix<scalar, NQ, 1, Options>;

    void integrate_impl(const Vec_t& x, const Vec_t& dx, Vec_t& out) const
    {
      out.noalias() = x + dx;
    }

    void diff_impl(const Vec_t& x0, const Vec_t& x1, Vec_t& out)
    const {
      out.noalias() = x1 - x0;
    }

  };

  }

