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
   * Wrap a Pinocchio Lie group into a ManifoldAbstract object.
   */
  template<typename _LieGroup>
  struct PinocchioLieGroup : public ManifoldAbstract<typename _LieGroup::Scalar>
  {
  public:
    using LieGroup = _LieGroup;
    using Self = PinocchioLieGroup<LieGroup>;
    using Scalar = typename LieGroup::Scalar;
    enum {
      Options = LieGroup::Options
    };
    using Base = ManifoldAbstract<Scalar>;
    LIENLP_DEFINE_MANIFOLD_TYPES(Base)

    LieGroup m_lg;
    PinocchioLieGroup() {}
    PinocchioLieGroup(const LieGroup& lg) : m_lg(lg) {}

    template<typename... Args>
    PinocchioLieGroup(Args... args) : m_lg(args...) {}

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

    inline int nx() const { return m_lg.nq(); }
    inline int ndx() const { return m_lg.nv(); }

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
  class MultibodyConfiguration : public ManifoldAbstract<_Scalar, _Options> {
  public:

    using Scalar = _Scalar;
    enum {
      Options = _Options
    };
    using Self = MultibodyConfiguration<Scalar, Options>;
    using PinModel = pin::ModelTpl<Scalar, Options>;
    using Base = ManifoldAbstract<Scalar, Options>;
    LIENLP_DEFINE_MANIFOLD_TYPES(Base)

    MultibodyConfiguration(const PinModel& model) : m_model(model)
    {};

    const PinModel& getModel() { return m_model; }

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

    inline int nx() const { return m_model.nq; }
    inline int ndx() const { return m_model.nv; }

    /// \}

  protected:
    const PinModel& m_model;

  };


}
