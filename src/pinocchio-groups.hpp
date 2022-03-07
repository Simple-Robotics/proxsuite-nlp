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
   * Wrap a Pinocchio Lie group
   */
  template<typename _LieGroup>
  struct PinocchioLieGroup : public ManifoldTpl<PinocchioLieGroup<_LieGroup>>
  {
    using LieGroup = _LieGroup;
    LieGroup m_lg;
    PinocchioLieGroup(LieGroup lg) : m_lg(lg) {}

    template<class P_t, class Tangent_t>
    void integrate_impl(const P_t& x,
                        const Tangent_t& v,
                        P_t& out) const
    {
      m_lg.integrate(x, v, out);
    }

    template<class P_t, class Tangent_t>
    void diff_impl(const P_t& x0,
                  const P_t& x1,
                  Tangent_t& out) const
    {
      m_lg.difference(x0, x1, out);
    }

    inline int nq_impl() const { return m_lg.nq(); }
    inline int nv_impl() const { return m_lg.nv(); }

  };

  template<typename LieGroup>
  struct traits<PinocchioLieGroup<LieGroup>>
  {
    using Self = PinocchioLieGroup<LieGroup>;
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

    template<class Vec_t, class Tangent_t>
    void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                        const Eigen::MatrixBase<Tangent_t>& v,
                        Eigen::MatrixBase<Vec_t>& out) const
    {
      pin::integrate(m_model, x, v, out);
    }

    template<class Vec_t, class Tangent_t>
    void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                  const Eigen::MatrixBase<Vec_t>& x1,
                  Eigen::MatrixBase<Tangent_t>& out) const
    {
      pin::difference(m_model, x0, x1, out);
    }

    inline int nq_impl() const { return m_model.nq; }
    inline int nv_impl() const { return m_model.nv; }

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
