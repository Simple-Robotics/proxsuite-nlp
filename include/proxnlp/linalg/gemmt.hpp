/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

namespace proxnlp {
namespace block_chol {
namespace backend {

template <typename Scalar, BlockKind LHS, BlockKind RHS> struct GemmT {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  static void fn(MatrixRef /*dst*/, ConstMatrixRef const & /*lhs*/,
                 ConstMatrixRef const & /*rhs*/, Scalar /*alpha*/) {}
};

template <typename Scalar> struct GemmT<Scalar, Diag, Diag> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is diagonal
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    auto v = lhs.diagonal().cwiseProduct(rhs.transpose().diagonal());
    isize n = v.rows();
    dst.diagonal().head(n) += alpha * v;
  }
};

template <typename Scalar> struct GemmT<Scalar, Diag, TriL> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is triu
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    // dst.template triangularView<Eigen::Upper>() +=
    // 		alpha * (lhs.diagonal().asDiagonal() *
    //              rhs.template triangularView<Eigen::Lower>().transpose());

    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).head(j + 1) += alpha * lhs.diagonal().cwiseProduct(
                                            rhs.transpose().col(j).head(j + 1));
    }
  }
};

template <typename Scalar> struct GemmT<Scalar, Diag, TriU> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is tril
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    // 		alpha * (lhs.diagonal().asDiagonal() *
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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    dst += alpha * (lhs.diagonal().asDiagonal() * rhs.transpose());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriL, Diag> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is tril
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    // 		alpha * (lhs.template triangularView<Eigen::Lower>() *
    //              rhs.diagonal().asDiagonal());

    isize m = dst.rows();
    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).tail(m - j) += (alpha * rhs(j, j)) * lhs.col(j).tail(m - j);
    }
  }
};

template <typename Scalar> struct GemmT<Scalar, TriL, TriL> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template
    //                 triangularView<Eigen::Upper>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriL, TriU> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is tril
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template
    //                 triangularView<Eigen::Lower>());
    dst.template triangularView<Eigen::Lower>() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriL, Dense> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    dst.noalias() +=
        lhs.template triangularView<Eigen::Lower>() * (alpha * rhs.transpose());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriU, Diag> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is triu
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    // 		alpha * (lhs.template triangularView<Eigen::Lower>() *
    //              rhs.diagonal().asDiagonal());

    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).head(j + 1) += (alpha * rhs(j, j)) * lhs.col(j).head(j + 1);
    }
  }
};

template <typename Scalar> struct GemmT<Scalar, TriU, TriL> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is triu
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    // PERF
    // dst.template triangularView<Eigen::Upper>() +=
    // 		alpha * (lhs.template triangularView<Eigen::Upper>() *
    //              rhs.transpose().triangularView<Eigen::Upper>());
    dst.template triangularView<Eigen::Upper>() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriU, TriU> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    // PERF
    // dst.noalias() += alpha * (lhs.template triangularView<Eigen::Upper>() *
    //                           rhs.transpose().template
    //                           triangularView<Eigen::Lower>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <typename Scalar> struct GemmT<Scalar, TriU, Dense> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // dst is dense
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    dst.noalias() += (lhs.template triangularView<Eigen::Upper>() *
                      (alpha * rhs.transpose()));
  }
};

template <typename Scalar> struct GemmT<Scalar, Dense, Diag> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    dst.noalias() += alpha * (lhs * rhs.transpose().diagonal().asDiagonal());
  }
};

template <typename Scalar> struct GemmT<Scalar, Dense, TriL> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <typename Scalar> struct GemmT<Scalar, Dense, TriU> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <typename Scalar> struct GemmT<Scalar, Dense, Dense> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  static void fn(MatrixRef dst, ConstMatrixRef const &lhs,
                 ConstMatrixRef const &rhs, Scalar alpha) {
    dst.noalias() += alpha * lhs * rhs.transpose();
  }
};

} // namespace backend
} // namespace block_chol
} // namespace proxnlp
