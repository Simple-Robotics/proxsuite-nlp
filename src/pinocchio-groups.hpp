#pragma once

#include "lienlp/manifold-base.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>


namespace lienlp {

namespace pin = pinocchio;

template<typename _Scalar, int _Options=0>
class PinocchioGroup : public ManifoldTpl<PinocchioGroup<_Scalar, _Options>> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using scalar = _Scalar;
  enum {
    Options = _Options
  };
  typedef pin::ModelTpl<_Scalar> PinModel;
  using Vec_t = typename PinModel::VectorXs;

  PinocchioGroup(const PinModel& model):
    m_model(model) {};

  void integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                      const Eigen::MatrixBase<Vec_t>& v,
                      Eigen::MatrixBase<Vec_t>& out) const
  {
    pin::integrate(m_model, x, v, out);
  }

  void diff_impl(const Eigen::MatrixBase<Vec_t>& x0,
                 const Eigen::MatrixBase<Vec_t>& x1,
                 Eigen::MatrixBase<Vec_t>& out) const
  {
    pin::difference(m_model, x0, x1, out);
  }

private:
  const PinModel m_model;

};

}
