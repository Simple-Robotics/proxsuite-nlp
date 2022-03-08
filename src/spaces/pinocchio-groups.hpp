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
    std::unique_ptr<LieGroup> m_lg;
    PinocchioLieGroup(LieGroup lg) : m_lg(std::make_unique<LieGroup>(lg)) {}

    /// \name Implementations

    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        Vec_t& out) const
    {
      m_lg->integrate(x.derived(), v.derived(), out);
    }

    template<class Vec_t, class Tangent_t>
    void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                   const Eigen::MatrixBase<Vec_t>& x1,
                   Eigen::MatrixBase<Tangent_t>& vout) const
    {
      m_lg->difference(x0.derived(), x1.derived(), vout);
    }

    template<class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate(const Eigen::MatrixBase<Vec_t>& x,
                    const Eigen::MatrixBase<Tangent_t>& v,
                    Eigen::MatrixBase<Jout_t>& Jx,
                    Eigen::MatrixBase<Jout_t>& Jv) const
    {
      m_lg->dIntegrate_dq(x.derived(), v.derived(), Jx.derived());
      m_lg->dIntegrate_dv(x.derived(), v.derived(), Jv.derived());
    }

    template<class Vec_t, class Jout_t>
    void Jdiff(const Eigen::MatrixBase<Vec_t>& x0,
               const Eigen::MatrixBase<Vec_t>& x1,
               Eigen::MatrixBase<Jout_t>& J0,
               Eigen::MatrixBase<Jout_t>& J1) const
    {
      m_lg->dDifference(x0.derived(), x1.derived(), J0.derived(), pin::ARG0);
      m_lg->dDifference(x0.derived(), x1.derived(), J1.derived(), pin::ARG1);
    }

    inline int nq_impl() const { return m_lg->nq(); }
    inline int nv_impl() const { return m_lg->nv(); }

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


  template<typename Scalar, int Options=0>
  class MultibodyConfiguration : public ManifoldTpl<MultibodyConfiguration<Scalar, Options>> {
  public:

    using Self = MultibodyConfiguration<Scalar, Options>;

    typedef pin::ModelTpl<Scalar> PinModel;

    MultibodyConfiguration(const PinModel& model) : m_model(model)
    {};

    /// \name implementations
    /// \{

    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        Eigen::MatrixBase<Vec_t>& out) const
    {
      pin::integrate(m_model, x.derived(), v.derived(), out);
    }

    template<class Vec_t, class Tangent_t, class Jout_t>
    void Jintegrate(const Eigen::MatrixBase<Vec_t>& x,
                    const Eigen::MatrixBase<Tangent_t>& v,
                    Eigen::MatrixBase<Jout_t>& Jx,
                    Eigen::MatrixBase<Jout_t>& Jv) const
    {
      pin::dIntegrate(m_model, x.derived(), v.derived(), Jx.derived(), pin::ARG0);
      pin::dIntegrate(m_model, x.derived(), v.derived(), Jv.derived(), pin::ARG1);
    }

    template<class Vec_t, class Tangent_t>
    void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                  const Eigen::MatrixBase<Vec_t>& x1,
                  Eigen::MatrixBase<Tangent_t>& vout) const
    {
      pin::difference(m_model, x0.derived(), x1.derived(), vout.derived());
    }

    template<class Vec_t, class Jout_t>
    void Jdiff(const Eigen::MatrixBase<Vec_t>& x0,
               const Eigen::MatrixBase<Vec_t>& x1,
               Eigen::MatrixBase<Jout_t>& J0,
               Eigen::MatrixBase<Jout_t>& J1) const
    {
      pin::dDifference(m_model, x0.derived(), x1.derived(), J0, pin::ARG0);
      pin::dDifference(m_model, x0.derived(), x1.derived(), J1, pin::ARG1);
    }

    inline int nq_impl() const { return m_model.nq; }
    inline int nv_impl() const { return m_model.nv; }

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
