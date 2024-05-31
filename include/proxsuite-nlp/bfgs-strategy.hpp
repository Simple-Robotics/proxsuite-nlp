/// @file
/// @copyright Copyright (C) 2024 INRIA
#pragma once

#include "proxsuite-nlp/fwd.hpp"

namespace proxsuite {
namespace nlp {

enum BFGSType { Hessian, InverseHessian };

// forward declaration
template <typename Scalar, BfgsType BFGS_TYPE> struct BfgsUpdateImpl;

template <typename Scalar, BfgsType BFGS_TYPE = BfgsType::InverseHessian>
class BFGSStrategy {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

public:
  explicit BfgsStrategy(const int num_vars)
      : M(Eigen::MatrixXd::Identity(num_vars, num_vars)), is_init(false),
        is_psd(true), x_prev(VectorXs::Zero(num_vars)),
        g_prev(VectorXs::Zero(num_vars)), s(VectorXs::Zero(num_vars)),
        y(VectorXs::Zero(num_vars)),
        xx_transpose(MatrixXs::Zero(num_vars, num_vars)),
        xy_transpose(MatrixXs::Zero(num_vars, num_vars)),
        V(MatrixXs::Zero(num_vars, num_vars)),
        VMinv(MatrixXs::Zero(num_vars, num_vars)),
        VMinvVt(MatrixXs::Zero(num_vars, num_vars)) {
    x_prev = VectorXs::Zero(num_vars);
    g_prev = VectorXs::Zero(num_vars);
  }

  void init(const ConstVectorRef &x0, const ConstVectorRef &g0) {
    x_prev = x0;
    g_prev = g0;
    is_init = true;
  }

  void update(const ConstVectorRef &x, const ConstVectorRef &g) {
    if (!is_init) {
      init(x, g);
      return;
    }
    PROXSUITE_NLP_NOMALLOC_BEGIN;
    s = x - x_prev;
    y = g - g_prev;
    const Scalar sy = s.dot(y);

    if (sy > 0) {
      BfgsUpdateImpl<Scalar, BFGS_TYPE>::update(*this, s, y);
      V.noalias() = MatrixXs::Identity(s.size(), s.size()) - xy_transpose / sy;
      VMinv.noalias() = V * M;
      VMinvVt.noalias() = VMinv * V.transpose();
      M = VMinvVt + xx_transpose / sy;
      is_psd = true;
    } else {
      is_psd = false;
      PROXSUITE_NLP_WARN("Skipping BFGS update as s^Ty <= 0");
    }
    x_prev = x;
    g_prev = g;
    PROXSUITE_NLP_NOMALLOC_END;
  }

public:
  MatrixXs M; // (inverse of the) Hessian approximation
  bool is_init;
  bool is_psd;

private:
  friend struct BfgsUpdateImpl<Scalar, BFGS_TYPE>;
  VectorXs x_prev; // previous iterate
  VectorXs g_prev; // previous gradient
  VectorXs s;      // delta iterate
  VectorXs y;      // delta gradient
  // temporary variables to avoid dynamic memory allocation
  MatrixXs xx_transpose;
  MatrixXs xy_transpose;
  MatrixXs V;
  MatrixXs VMinv;
  MatrixXs VMinvVt;
};

// Specialization of update_impl method for BfgsType::InverseHessian
// see Nocedal and Wright, Numerical Optimization, 2nd ed., p. 140, eqn 6.17
// (BFGS update)
template <typename Scalar>
struct BfgsUpdateImpl<Scalar, BfgsType::InverseHessian> {
  static void update(BfgsStrategy<Scalar, BfgsType::InverseHessian> &strategy,
                     const typename BfgsStrategy<
                         Scalar, BfgsType::InverseHessian>::ConstVectorRef &s,
                     const typename BfgsStrategy<
                         Scalar, BfgsType::InverseHessian>::ConstVectorRef &y) {
    strategy.xx_transpose.noalias() = s * s.transpose();
    strategy.xy_transpose.noalias() = s * y.transpose();
  }
};

// Specialization of update_impl method for BfgsType::Hessian
// see Nocedal and Wright, Numerical Optimization, 2nd ed., p. 139, eqn 6.13
// (DFP update)
template <typename Scalar> struct BfgsUpdateImpl<Scalar, BfgsType::Hessian> {
  static void update(
      BfgsStrategy<Scalar, BfgsType::Hessian> &strategy,
      const typename BfgsStrategy<Scalar, BfgsType::Hessian>::ConstVectorRef &s,
      const typename BfgsStrategy<Scalar, BfgsType::Hessian>::ConstVectorRef
          &y) {
    strategy.xx_transpose.noalias() = y * y.transpose();
    strategy.xy_transpose.noalias() = y * s.transpose();
  }
};

} // namespace nlp
} // namespace proxsuite
