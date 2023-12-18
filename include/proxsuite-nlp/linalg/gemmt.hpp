/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/linalg/block-kind.hpp"

namespace proxsuite {
namespace nlp {
namespace linalg {
namespace backend {

#define PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha)            \
  template <typename Dst, typename Lhs, typename Rhs>                          \
  static void fn(Eigen::MatrixBase<Dst> &dst,                                  \
                 Eigen::MatrixBase<Lhs> const &lhs,                            \
                 Eigen::MatrixBase<Rhs> const &rhs, Scalar alpha)

template <typename Scalar, BlockKind LHS, BlockKind RHS> struct GemmT {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, /*dst*/, /*lhs*/, /*rhs*/, /*alpha*/) {}
};

template <typename Scalar> struct GemmT<Scalar, Diag, Diag> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is diagonal
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    auto v = lhs.diagonal().cwiseProduct(rhs.transpose().diagonal());
    isize n = v.rows();
    dst.diagonal().head(n) += alpha * v;
  }
};

template <typename Scalar> struct GemmT<Scalar, Diag, TriL> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is triu
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    // dst.template triangularView<Eigen::Upper>() +=
    //     alpha * (lhs.diagonal().asDiagonal() *
    //              rhs.template triangularView<Eigen::Lower>().transpose());

    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).head(j + 1) += alpha * lhs.diagonal().cwiseProduct(
                                            rhs.transpose().col(j).head(j + 1));
    }
  }
};

template <typename Scalar> struct GemmT<Scalar, Diag, TriU> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is tril
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    //     alpha * (lhs.diagonal().asDiagonal() *
    //              rhs.template triangularView<Eigen::Upper>().transpose());

    isize m = dst.rows();
    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).tail(m - j) += alpha * lhs.diagonal().cwiseProduct(
                                            rhs.transpose().col(j).tail(m - j));
    }
  }
};

template <typename Scalar> struct GemmT<Scalar, Diag, Dense> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    dst += alpha * (lhs.diagonal().asDiagonal() * rhs.transpose());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriL, Diag> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is tril
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    //     alpha * (lhs.template triangularView<Eigen::Lower>() *
    //              rhs.diagonal().asDiagonal());

    isize m = dst.rows();
    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).tail(m - j) += (alpha * rhs(j, j)) * lhs.col(j).tail(m - j);
    }
  }
};

template <typename Scalar> struct GemmT<Scalar, TriL, TriL> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template
    //                 triangularView<Eigen::Upper>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriL, TriU> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is tril
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template
    //                 triangularView<Eigen::Lower>());
    dst.template triangularView<Eigen::Lower>() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriL, Dense> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    dst.noalias() +=
        lhs.template triangularView<Eigen::Lower>() * (alpha * rhs.transpose());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriU, Diag> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is triu
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    //     alpha * (lhs.template triangularView<Eigen::Lower>() *
    //              rhs.diagonal().asDiagonal());

    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).head(j + 1) += (alpha * rhs(j, j)) * lhs.col(j).head(j + 1);
    }
  }
};

template <typename Scalar> struct GemmT<Scalar, TriU, TriL> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is triu
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    // PERF
    // dst.template triangularView<Eigen::Upper>() +=
    //     alpha * (lhs.template triangularView<Eigen::Upper>() *
    //              rhs.transpose().triangularView<Eigen::Upper>());
    dst.template triangularView<Eigen::Upper>() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriU, TriU> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    // PERF
    // dst.noalias() += alpha * (lhs.template triangularView<Eigen::Upper>() *
    //                           rhs.transpose().template
    //                           triangularView<Eigen::Lower>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriU, Dense> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    dst.noalias() += (lhs.template triangularView<Eigen::Upper>() *
                      (alpha * rhs.transpose()));
  }
};

template <typename Scalar> struct GemmT<Scalar, Dense, Diag> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    dst.noalias() += alpha * (lhs * rhs.transpose().diagonal().asDiagonal());
  }
};

template <typename Scalar> struct GemmT<Scalar, Dense, TriL> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <typename Scalar> struct GemmT<Scalar, Dense, TriU> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <typename Scalar> struct GemmT<Scalar, Dense, Dense> {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  PROXSUITE_NLP_GEMMT_SIGNATURE(Scalar, dst, lhs, rhs, alpha) {
    dst.noalias() += alpha * lhs * rhs.transpose();
  }
};

template <typename Scalar, typename DstDerived, typename LhsDerived,
          typename RhsDerived>
inline void gemmt(Eigen::MatrixBase<DstDerived> &dst,
                  Eigen::MatrixBase<LhsDerived> const &lhs,
                  Eigen::MatrixBase<RhsDerived> const &rhs, BlockKind lhs_kind,
                  BlockKind rhs_kind, Scalar alpha) {
  // dst += alpha * lhs * rhs.T
  switch (lhs_kind) {
  case Zero: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Scalar, Zero, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Scalar, Zero, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Scalar, Zero, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Scalar, Zero, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Scalar, Zero, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case Diag: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Scalar, Diag, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Scalar, Diag, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Scalar, Diag, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Scalar, Diag, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Scalar, Diag, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case TriL: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Scalar, TriL, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Scalar, TriL, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Scalar, TriL, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Scalar, TriL, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Scalar, TriL, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case TriU: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Scalar, TriU, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Scalar, TriU, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Scalar, TriU, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Scalar, TriU, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Scalar, TriU, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case Dense: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Scalar, Dense, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Scalar, Dense, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Scalar, Dense, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Scalar, Dense, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Scalar, Dense, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  }
}

} // namespace backend
} // namespace linalg
} // namespace nlp
} // namespace proxsuite
