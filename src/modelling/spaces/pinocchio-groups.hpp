#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/manifold-base.hpp"

#include <pinocchio/multibody/liegroup/liegroup-base.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <memory>


namespace proxnlp
{

  namespace pin = pinocchio;

  /**
   * Wrap a Pinocchio Lie group into a ManifoldAbstractTpl object.
   */
  template<typename _LieGroup>
  struct PinocchioLieGroup : public ManifoldAbstractTpl<typename _LieGroup::Scalar>
  {
  public:
    using LieGroup = _LieGroup;
    using Self = PinocchioLieGroup<LieGroup>;
    using Scalar = typename LieGroup::Scalar;
    enum {
      Options = LieGroup::Options
    };
    using Base = ManifoldAbstractTpl<Scalar>;
    PROXNLP_DEFINE_MANIFOLD_TYPES(Base)

    LieGroup m_lg;
    PinocchioLieGroup() {}
    PinocchioLieGroup(const LieGroup& lg) : m_lg(lg) {}

    template<typename... Args>
    PinocchioLieGroup(Args... args) : m_lg(args...) {}

    inline int nx() const { return m_lg.nq(); }
    inline int ndx() const { return m_lg.nv(); }

    /// \name Implementations

    void integrate_impl(const ConstVectorRef& x,
                   const ConstVectorRef& v,
                   VectorRef out) const
    {
      m_lg.integrate(x, v, out);
    }

    void difference_impl(const ConstVectorRef& x0,
                    const ConstVectorRef& x1,
                    VectorRef vout) const
    {
      m_lg.difference(x0, x1, vout);
    }

    void Jintegrate_impl(const ConstVectorRef& x,
                    const ConstVectorRef& v,
                    MatrixRef Jout,
                    int arg) const
    {
      switch (arg) {
        case 0:
          m_lg.dIntegrate_dq(x, v, Jout);
          break;
        case 1:
          m_lg.dIntegrate_dv(x, v, Jout);
          break;
      }
    }

    void Jdifference_impl(const ConstVectorRef& x0,
                     const ConstVectorRef& x1,
                     MatrixRef Jout,
                     int arg) const
    {
      switch (arg) {
        case 0:
          m_lg.dDifference(x0, x1, Jout, pin::ARG0);
          break;
        case 1:
          m_lg.dDifference(x0, x1, Jout, pin::ARG1);
          break;
      }
    }

    virtual void interpolate_impl(const ConstVectorRef& x0,
                             const ConstVectorRef& x1,
                             const Scalar& u,
                             VectorRef out) const
    {
      m_lg.interpolate(x0, x1, u, out);
    }

    PointType neutral() const
    {
      return m_lg.neutral();
    }

    PointType rand() const
    {
      return m_lg.random();
    }

  };

}
