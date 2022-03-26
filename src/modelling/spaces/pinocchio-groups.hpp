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
   * Wrap a Pinocchio Lie group into a ManifoldTpl object.
   */
  template<typename _LieGroup>
  struct PinocchioLieGroup : public ManifoldTpl<PinocchioLieGroup<_LieGroup>>
  {
    using LieGroup = _LieGroup;
    using Self = PinocchioLieGroup<LieGroup>;
    LIENLP_DEFINE_INTERFACE_TYPES(Self)

    LieGroup m_lg;
    PinocchioLieGroup() {}
    PinocchioLieGroup(const LieGroup& lg) : m_lg(lg) {}

    template<typename... Args>
    PinocchioLieGroup(Args... args) : m_lg(args...) {}

    /// \name Implementations

    template<class Vec_t, class Tangent_t, class Out_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        const Eigen::MatrixBase<Out_t>& out) const
    {
      m_lg.integrate(x.derived(), v.derived(), out.derived());
    }

    template<class Vec1_t, class Vec2_t, class Tangent_t>
    void difference_impl(const Eigen::MatrixBase<Vec1_t>& x0,
                         const Eigen::MatrixBase<Vec2_t>& x1,
                         const Eigen::MatrixBase<Tangent_t>& vout) const
    {
      m_lg.difference(x0.derived(), x1.derived(), vout.derived());
    }

    template<int arg, class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                         const Eigen::MatrixBase<Tangent_t>& v,
                         const Eigen::MatrixBase<Jout_t>& Jout) const
    {
      switch (arg) {
        case 0:
          m_lg.dIntegrate_dq(x.derived(), v.derived(), Jout.derived());
          break;
        case 1:
          m_lg.dIntegrate_dv(x.derived(), v.derived(), Jout.derived());
          break;
      }
    }

    template<int arg, class Vec1_t, class Vec2_t, class Jout_t>
    void Jdifference_impl(const Eigen::MatrixBase<Vec1_t>& x0,
                          const Eigen::MatrixBase<Vec2_t>& x1,
                          const Eigen::MatrixBase<Jout_t>& Jout) const
    {
      switch (arg) {
        case 0:
          m_lg.dDifference(x0.derived(), x1.derived(), Jout.derived(), pin::ARG0);
          break;
        case 1:
          m_lg.dDifference(x0.derived(), x1.derived(), Jout.derived(), pin::ARG1);
          break;
      }
    }

    inline int nx_impl() const { return m_lg.nq(); }
    inline int ndx_impl() const { return m_lg.nv(); }

    PointType neutral_impl() const
    {
      return m_lg.neutral();
    }

    PointType rand_impl() const
    {
      return m_lg.random();
    }

  };

  template<typename LieGroup>
  struct traits<PinocchioLieGroup<LieGroup>>
  {
    using Scalar = typename LieGroup::Scalar;
    enum {
      NQ = LieGroup::NQ,
      NV = LieGroup::NV,
      Options = LieGroup::Options
    };

  };


  template<typename _Scalar, int Options=0>
  class MultibodyConfiguration : public ManifoldTpl<MultibodyConfiguration<_Scalar, Options>> {
  public:

    using Self = MultibodyConfiguration<_Scalar, Options>;
    LIENLP_DEFINE_INTERFACE_TYPES(Self)

    using PinModel = pin::ModelTpl<Scalar, Options>;

    MultibodyConfiguration(const PinModel& model) : m_model(model)
    {};

    const PinModel& getModel() { return m_model; }

    PointType neutral_impl() const
    {
      return pinocchio::neutral(m_model);
    }

    PointType rand_impl() const
    {
      return pinocchio::randomConfiguration(m_model);
    }

    /// \name implementations
    /// \{

    template<class Vec_t, class Tangent_t, class Out_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        const Eigen::MatrixBase<Out_t>& xout) const
    {
      pin::integrate(m_model, x.derived(), v.derived(), xout.derived());
    }

    template<int arg, class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                         const Eigen::MatrixBase<Tangent_t>& v,
                         const Eigen::MatrixBase<Jout_t>& Jout) const
    {
      switch (arg) {
        case 0:
          pin::dIntegrate(m_model, x.derived(), v.derived(), Jout.derived(), pin::ARG0);
          break;
        case 1:
          pin::dIntegrate(m_model, x.derived(), v.derived(), Jout.derived(), pin::ARG1);
          break;
      }
    }

    template<class Vec1_t, class Vec2_t, class Out_t>
    void difference_impl(const Eigen::MatrixBase<Vec1_t>& x0,
                         const Eigen::MatrixBase<Vec2_t>& x1,
                         const Eigen::MatrixBase<Out_t>& vout) const
    {
      pin::difference(m_model, x0.derived(), x1.derived(), vout.derived());
    }

    template<int arg, class Vec1_t, class Vec2_t, class Jout_t>
    void Jdifference_impl(const Eigen::MatrixBase<Vec1_t>& x0,
                          const Eigen::MatrixBase<Vec2_t>& x1,
                          const Eigen::MatrixBase<Jout_t>& Jout) const
    {
      switch (arg) {
        case 0:
          pin::dDifference(m_model, x0.derived(), x1.derived(), Jout.derived(), pin::ARG0);
          break;
        case 1:
          pin::dDifference(m_model, x0.derived(), x1.derived(), Jout.derived(), pin::ARG1);
          break;
      }
    }

    inline int nx_impl() const { return m_model.nq; }
    inline int ndx_impl() const { return m_model.nv; }

    /// \}

  protected:
    const PinModel& m_model;

  };


  template<typename scalar, int options>
  struct traits<MultibodyConfiguration<scalar, options>>
  {
    using Scalar = scalar;
    enum {
      NQ = -1,
      NV = -1,
      Options = options
    };
  };


}
