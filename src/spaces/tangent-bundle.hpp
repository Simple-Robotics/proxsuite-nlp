#pragma once

#include "lienlp/manifold-base.hpp"

namespace lienlp
{

  /**
   * @brief     Tangent bundle of a base manifold M. This construction is recursive.
   */
  template<class Base>
  struct TangentBundle<Base> : public ManifoldTpl<TangentBundle<Base>>
  {
    TangentBundle<Base>(Base* base) : baseManifold(base) {}; 
  protected:
    Base* baseManifold;
  public:
    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& dx,
                        Eigen::MatrixBase<Vec_t>& out) const
    {
      const int nq_ = base->nx();
      const int nv_ = base->ndx();
      Base::Point_t base_pt = x.head(nq_);
      base->integrate(base_pt, dx.head(nv_), out.head(nq_));
      out.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
    }

    template<class Vec_t, class Tangent_t>
    void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                   const Eigen::MatrixBase<Vec_t>& x1,
                   Eigen::MatrixBase<Tangent_t>& out) const
    {
      const int nq_ = base->nx();
      const int nv_ = base->ndx();
      base->diff(x0.head(nq_), x1.head(nq_), out.head(nv_));
      out.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
    }

    inline int nx_impl() const { return base->nx() + base->ndx(); }
    inline int ndx_impl() const { return 2 * base->ndx(); }

    /// TODO implement Jintegrate, Jdiff

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
