#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/manifold-base.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/liegroup/liegroup-base.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <memory>


namespace lienlp {

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
    PinocchioLieGroup(LieGroup lg) : m_lg(lg) {}

    /// \name Implementations

    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        Vec_t& out) const
    {
      m_lg.integrate(x.derived(), v.derived(), out);
    }

    template<class Vec_t, class Tangent_t>
    void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                   const Eigen::MatrixBase<Vec_t>& x1,
                   Eigen::MatrixBase<Tangent_t>& vout) const
    {
      m_lg.difference(x0.derived(), x1.derived(), vout);
    }

    template<int arg, class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                    const Eigen::MatrixBase<Tangent_t>& v,
                    Eigen::MatrixBase<Jout_t>& Jout) const
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

    template<int arg, class Vec_t, class Jout_t>
    void Jdiff_impl(const Eigen::MatrixBase<Vec_t>& x0,
               const Eigen::MatrixBase<Vec_t>& x1,
               Eigen::MatrixBase<Jout_t>& Jout) const
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

    Point_t zero_impl() const
    {
      return m_lg.neutral();
    }

    Point_t rand_impl() const
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

    typedef pin::ModelTpl<Scalar, Options> PinModel;

    MultibodyConfiguration(const PinModel& model) : m_model(model)
    {};

    Point_t zero_impl() const
    {
      return pinocchio::neutral(m_model);
    }

    Point_t rand_impl() const
    {
      return pinocchio::randomConfiguration(m_model);
    }

    /// \name implementations
    /// \{

    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        Eigen::MatrixBase<Vec_t>& xout) const
    {
      pin::integrate(m_model, x.derived(), v.derived(), xout.derived());
    }

    template<int arg, class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                    const Eigen::MatrixBase<Tangent_t>& v,
                    Eigen::MatrixBase<Jout_t>& Jout) const
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

    template<class Vec_t, class Tangent_t>
    void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                   const Eigen::MatrixBase<Vec_t>& x1,
                   Eigen::MatrixBase<Tangent_t>& vout) const
    {
      pin::difference(m_model, x0.derived(), x1.derived(), vout.derived());
    }

    template<int arg, class Vec_t, class Jout_t>
    void Jdiff_impl(const Eigen::MatrixBase<Vec_t>& x0,
               const Eigen::MatrixBase<Vec_t>& x1,
               Eigen::MatrixBase<Jout_t>& Jout) const
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
    const PinModel m_model;

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
