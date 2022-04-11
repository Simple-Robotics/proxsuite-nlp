#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/manifold-base.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/liegroup/liegroup-base.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <memory>


namespace lienlp
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
    LIENLP_DEFINE_MANIFOLD_TYPES(Base)

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

  template<typename _Scalar, int _Options=0>
  class MultibodyConfiguration : public ManifoldAbstractTpl<_Scalar, _Options> {
  public:

    using Scalar = _Scalar;
    enum {
      Options = _Options
    };
    using Self = MultibodyConfiguration<Scalar, Options>;
    using ModelType = pin::ModelTpl<Scalar, Options>;
    using Base = ManifoldAbstractTpl<Scalar, Options>;
    LIENLP_DEFINE_MANIFOLD_TYPES(Base)

    MultibodyConfiguration(const ModelType& model)
      : m_model(model)
      {};

    const ModelType& getModel() { return m_model; }

    PointType neutral() const
    {
      return pinocchio::neutral(m_model);
    }

    PointType rand() const
    {
      return pinocchio::randomConfiguration(m_model);
    }

    /// \name implementations
    /// \{

    void integrate_impl(const ConstVectorRef& x,
                        const ConstVectorRef& v,
                        VectorRef xout) const
    {
      pin::integrate(m_model, x, v, xout);
    }

    void Jintegrate_impl(const ConstVectorRef& x,
                    const ConstVectorRef& v,
                    MatrixRef Jout,
                    int arg) const
    {
      switch (arg) {
        case 0:
          pin::dIntegrate(m_model, x, v, Jout, pin::ARG0);
          break;
        case 1:
          pin::dIntegrate(m_model, x, v, Jout, pin::ARG1);
          break;
      }
    }

    void difference_impl(const ConstVectorRef& x0,
                    const ConstVectorRef& x1,
                    VectorRef vout) const
    {
      pin::difference(m_model, x0, x1, vout);
    }

    void Jdifference_impl(const ConstVectorRef& x0,
                     const ConstVectorRef& x1,
                     MatrixRef Jout,
                     int arg) const
    {
      switch (arg) {
        case 0:
          pin::dDifference(m_model, x0, x1, Jout, pin::ARG0);
          break;
        case 1:
          pin::dDifference(m_model, x0, x1, Jout, pin::ARG1);
          break;
      }
    }

    virtual void interpolate_impl(const ConstVectorRef& x0,
                             const ConstVectorRef& x1,
                             const Scalar& u,
                             VectorRef out) const
    {
      pin::interpolate(m_model, x0, x1, u, out);
    }

    inline int nx() const { return m_model.nq; }
    inline int ndx() const { return m_model.nv; }

    /// \}

  protected:
    const ModelType& m_model;

  };


}
