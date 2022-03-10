#pragma once

#include "lienlp/manifold-base.hpp"

namespace lienlp
{

  /**
   * @brief     Tangent bundle of a base manifold M. This construction is recursive.
   */
  template<class Base>
  struct TangentBundle : public ManifoldTpl<TangentBundle<Base>>
  {
  protected:
    Base m_base;
  public:
    using Self = TangentBundle<Base>;

    using Scalar = typename traits<Self>::Scalar;
    enum {
      NQ = traits<Self>::NQ,
      NV = traits<Self>::NV,
      Options = traits<Self>::Options
    };

    using Point_t = Eigen::Matrix<Scalar, NQ, 1, Options>;
    using TangentVec_t = Eigen::Matrix<Scalar, NV, 1, Options>;
    using Jac_t = Eigen::Matrix<Scalar, NV, NV, Options>; 

    TangentBundle<Base>(Base base) : m_base(base) {}; 

    /// Declare implementations

    inline int nx_impl() const { return m_base.nx() + m_base.ndx(); }
    inline int ndx_impl() const { return 2 * m_base.ndx(); }

    Point_t zero_impl() const;
    Point_t rand_impl() const;


    /// @name   Implementations

    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& dx,
                        Eigen::MatrixBase<Vec_t>& out) const
    {
      const int nq_ = m_base.nx();
      const int nv_ = m_base.ndx();
      m_base.integrate(x.head(nq_), dx.head(nv_), out.head(nq_));
      out.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
    }

    template<class Vec_t, class Tangent_t>
    void difference_impl(const Eigen::MatrixBase<Vec_t>& x0,
                   const Eigen::MatrixBase<Vec_t>& x1,
                   Eigen::MatrixBase<Tangent_t>& out) const
    {
      const int nq_ = m_base.nx();
      const int nv_ = m_base.ndx();
      m_base.difference(x0.head(nq_), x1.head(nq_), out.head(nv_));
      out.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
    }

    // template<class Vec_t, class Tangent_t>
    /// TODO implement Jintegrate_impl, Jdifference_impl

  };

  template<class M>
  struct traits<TangentBundle<M>>
  {
    using base_traits = traits<M>;
    using Scalar = typename base_traits::Scalar;
    enum {
      NQ = Eigen::Dynamic,
      NV = Eigen::Dynamic,
      Options = base_traits::Options
    };
  };

} // namespace lienlp

#include "lienlp/spaces/tangent-bundle.hxx"
