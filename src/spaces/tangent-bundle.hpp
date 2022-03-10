#pragma once

#include "lienlp/macros.hpp"
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


    /// @name   Implementations of operators

    template<class Vec_t, class Tangent_t, class Out_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& dx,
                        const Eigen::MatrixBase<Out_t>& out) const;

    template<class Vec_t, class Tangent_t>
    void difference_impl(const Eigen::MatrixBase<Vec_t>& x0,
                   const Eigen::MatrixBase<Vec_t>& x1,
                   const Eigen::MatrixBase<Tangent_t>& out) const;

    template<int arg, class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                         const Eigen::MatrixBase<Tangent_t>& v,
                         const Eigen::MatrixBase<Jout_t>& Jout) const;

    template<int arg, class Vec_t, class Jout_t>
    void Jdifference_impl(const Eigen::MatrixBase<Vec_t>& x0,
                          const Eigen::MatrixBase<Vec_t>& x1,
                          const Eigen::MatrixBase<Jout_t>& Jout) const;

    /// Get base point of an element of the tangent bundle.
    /// This map is exactly the natural projection.
    template<typename Point>
    auto getBasePoint(const Eigen::MatrixBase<Point>& x) const
    {
      return x.derived().template head<Base::NQ>(m_base.nx());
    }

    template<typename Point>
    auto getBasePointWrite(const Eigen::MatrixBase<Point>& x) const
    {
      return LIENLP_EIGEN_CONST_CAST(Point, x).template head<Base::NQ>(m_base.nx());
    }

    template<typename Tangent>
    auto getBaseTangent(const Eigen::MatrixBase<Tangent>& v) const
    {
      return v.derived().template head<Base::NV>(m_base.ndx());
    }

    template<typename Tangent>
    auto getTangentHeadWrite(const Eigen::MatrixBase<Tangent>& v) const
    {
      return LIENLP_EIGEN_CONST_CAST(Tangent, v).template head<Base::NV>(m_base.ndx());
    }

    template<typename Jac>
    Eigen::Block<Jac, Base::NV, Base::NV>
    getBaseJacobian(const Eigen::MatrixBase<Jac>& J) const
    {
      return LIENLP_EIGEN_CONST_CAST(Jac, J).template topLeftCorner<Base::NV,Base::NV>(m_base.ndx(), m_base.ndx());
    }

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
