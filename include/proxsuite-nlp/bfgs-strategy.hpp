/// @file
/// @copyright Copyright (C) 2024-2025 INRIA
#pragma once
#include "proxsuite-nlp/fwd.hpp"

namespace proxsuite {
namespace nlp {

enum BFGSType { Hessian, InverseHessian };

// forward declaration
template <typename Scalar, BFGSType BFGS_TYPE> struct BFGSUpdateImpl;

template <typename Scalar, BFGSType BFGS_TYPE = BFGSType::InverseHessian>
class BFGSStrategy {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);

public:
  BFGSStrategy()
      : is_init(false), is_psd(false), x_prev(), g_prev(), s(), y(),
        xx_transpose(), xy_transpose(), V(), VMinv(), VMinvVt(),
        is_valid(false) {}

  explicit BFGSStrategy(const int num_vars)
      : is_init(false), is_psd(true), x_prev(VectorXs::Zero(num_vars)),
        g_prev(VectorXs::Zero(num_vars)), s(VectorXs::Zero(num_vars)),
        y(VectorXs::Zero(num_vars)),
        xx_transpose(MatrixXs::Zero(num_vars, num_vars)),
        xy_transpose(MatrixXs::Zero(num_vars, num_vars)),
        V(MatrixXs::Zero(num_vars, num_vars)),
        VMinv(MatrixXs::Zero(num_vars, num_vars)),
        VMinvVt(MatrixXs::Zero(num_vars, num_vars)), is_valid(true) {
    x_prev = VectorXs::Zero(num_vars);
    g_prev = VectorXs::Zero(num_vars);
  }

  void init(const ConstVectorRef &x0, const ConstVectorRef &g0) {
    if (!is_valid) {
      throw std::runtime_error("Cannot initialize an invalid BFGSStrategy. Use "
                               "the constructor with num_vars first.");
    }

    x_prev = x0;
    g_prev = g0;
    is_init = true;
  }

  void update(const ConstVectorRef &x, const ConstVectorRef &g,
              MatrixXs &hessian) {
    if (!is_init) {
      init(x, g);
      return;
    }
    PROXSUITE_NLP_NOMALLOC_BEGIN;
    s = x - x_prev;
    y = g - g_prev;
    const Scalar sy = s.dot(y);

    if (sy > 0) {
      BFGSUpdateImpl<Scalar, BFGS_TYPE>::update(*this, s, y);
      V.noalias() = MatrixXs::Identity(s.size(), s.size()) - xy_transpose / sy;
      VMinv.noalias() = V * hessian;
      VMinvVt.noalias() = VMinv * V.transpose();
      hessian = VMinvVt + xx_transpose / sy;
      is_psd = true;
    } else {
      is_psd = false;
      PROXSUITE_NLP_WARN("Skipping BFGS update as s^Ty <= 0");
    }
    x_prev = x;
    g_prev = g;
    PROXSUITE_NLP_NOMALLOC_END;
  }
  bool isValid() const { return is_valid; }

public:
  bool is_init;
  bool is_psd;

private:
  friend struct BFGSUpdateImpl<Scalar, BFGS_TYPE>;
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
  bool is_valid;
};

// Specialization of update_impl method for BFGSType::InverseHessian
// see Nocedal and Wright, Numerical Optimization, 2nd ed., p. 140, eqn 6.17
// (BFGS update)
template <typename Scalar>
struct BFGSUpdateImpl<Scalar, BFGSType::InverseHessian> {
  static void update(BFGSStrategy<Scalar, BFGSType::InverseHessian> &strategy,
                     const typename BFGSStrategy<
                         Scalar, BFGSType::InverseHessian>::ConstVectorRef &s,
                     const typename BFGSStrategy<
                         Scalar, BFGSType::InverseHessian>::ConstVectorRef &y) {
    strategy.xx_transpose.noalias() = s * s.transpose();
    strategy.xy_transpose.noalias() = s * y.transpose();
  }
};

// Specialization of update_impl method for BFGSType::Hessian
// see Nocedal and Wright, Numerical Optimization, 2nd ed., p. 139, eqn 6.13
// (DFP update)
template <typename Scalar> struct BFGSUpdateImpl<Scalar, BFGSType::Hessian> {
  static void update(
      BFGSStrategy<Scalar, BFGSType::Hessian> &strategy,
      const typename BFGSStrategy<Scalar, BFGSType::Hessian>::ConstVectorRef &s,
      const typename BFGSStrategy<Scalar, BFGSType::Hessian>::ConstVectorRef
          &y) {
    strategy.xx_transpose.noalias() = y * y.transpose();
    strategy.xy_transpose.noalias() = y * s.transpose();
  }
};

} // namespace nlp
} // namespace proxsuite
