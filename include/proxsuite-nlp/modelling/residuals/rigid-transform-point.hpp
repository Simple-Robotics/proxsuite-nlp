#pragma once

#include "proxsuite-nlp/function-base.hpp"
#include "proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp"
#include <pinocchio/spatial/se3-tpl.hpp>

namespace proxnlp {

template <typename Scalar>
struct RigidTransformationPointActionTpl : C2FunctionTpl<Scalar> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // convenience for rigid-body transformations
  using SE3 = pin::SE3Tpl<Scalar>;
  using QuatConstMap = Eigen::Map<const typename SE3::Quaternion>;
  using Base = C2FunctionTpl<Scalar>;
  using Matrix33s = Eigen::Matrix<Scalar, 3, 3>;

  SETpl<3, Scalar> space_; // manifold
  Vector3s point_;

  RigidTransformationPointActionTpl(const Eigen::Ref<const Vector3s> &point)
      : Base(7, 6, 3), space_(), point_(point), skew_point_(pin::skew(point)) {}

  VectorXs operator()(const ConstVectorRef &x) const override {
    QuatConstMap q(x.template tail<4>().data());
    SE3 M(q, x.template head<3>());

    return M.actOnEigenObject(point_);
  }

  void computeJacobian(const ConstVectorRef &x, MatrixRef Jout) const override {
    assert(Jout.rows() == 3 && Jout.cols() == 6);
    QuatConstMap q(x.template tail<4>().data());

    Jout.template leftCols<3>() = q.matrix();
    Jout.template rightCols<3>().noalias() = -q.matrix() * skew_point_;
  }

  Eigen::Ref<const Matrix33s> skew_point() const { return skew_point_; }

private:
  Matrix33s skew_point_;
};

} // namespace proxnlp

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/modelling/residuals/rigid-transform-point.hpp"
#endif
