#pragma once

template <BlockKind LHS, BlockKind RHS> struct GemmT;

template <BlockKind LHS> struct GemmT<LHS, Zero> {
  static void fn(MatrixRef const & /*dst*/, MatrixRef const & /*lhs*/,
                 MatrixRef const & /*rhs*/, Scalar /*alpha*/) {}
};
template <BlockKind RHS> struct GemmT<Zero, RHS> {
  static void fn(MatrixRef const & /*dst*/, MatrixRef const & /*lhs*/,
                 MatrixRef const & /*rhs*/, Scalar /*alpha*/) {}
};
template <> struct GemmT<Zero, Zero> {
  static void fn(MatrixRef const & /*dst*/, MatrixRef const & /*lhs*/,
                 MatrixRef const & /*rhs*/, Scalar /*alpha*/) {}
};

template <> struct GemmT<Diag, Diag> {
  // dst is diagonal
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    auto v = lhs.diagonal().cwiseProduct(rhs.transpose().diagonal());
    isize n = v.rows();
    dst.diagonal().head(n) += alpha * v;
  }
};

template <> struct GemmT<Diag, TriL> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
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

template <> struct GemmT<Diag, TriU> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
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

template <> struct GemmT<Diag, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    dst += alpha * (lhs.diagonal().asDiagonal() * rhs.transpose());
  }
};

template <> struct GemmT<TriL, Diag> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
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

template <> struct GemmT<TriL, TriL> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template triangularView<Eigen::Upper>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <> struct GemmT<TriL, TriU> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template triangularView<Eigen::Lower>());
    dst.triangularView<Eigen::Lower>() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <> struct GemmT<TriL, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    dst.noalias() +=
        lhs.template triangularView<Eigen::Lower>() * (alpha * rhs.transpose());
  }
};

template <> struct GemmT<TriU, Diag> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    // dst.template triangularView<Eigen::Lower>() +=
    // 		alpha * (lhs.template triangularView<Eigen::Lower>() *
    //              rhs.diagonal().asDiagonal());

    isize n = dst.cols();

    for (isize j = 0; j < n; ++j) {
      dst.col(j).head(j + 1) += (alpha * rhs(j, j)) * lhs.col(j).head(j + 1);
    }
  }
};

template <> struct GemmT<TriU, TriL> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    // PERF
    // dst.template triangularView<Eigen::Upper>() +=
    // 		alpha * (lhs.template triangularView<Eigen::Upper>() *
    //              rhs.transpose().triangularView<Eigen::Upper>());
    dst.template triangularView<Eigen::Upper>() +=
        alpha * (lhs * rhs.transpose().triangularView<Eigen::Upper>());
  }
};

template <> struct GemmT<TriU, TriU> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    // PERF
    // dst.noalias() += alpha * (lhs.template triangularView<Eigen::Upper>() *
    //                           rhs.transpose().template
    //                           triangularView<Eigen::Lower>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <> struct GemmT<TriU, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    dst.noalias() += (lhs.template triangularView<Eigen::Upper>() *
                      (alpha * rhs.transpose()));
  }
};

template <> struct GemmT<Dense, Diag> {
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    dst.noalias() += alpha * (lhs * rhs.transpose().diagonal().asDiagonal());
  }
};

template <> struct GemmT<Dense, TriL> {
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().triangularView<Eigen::Upper>());
  }
};

template <> struct GemmT<Dense, TriU> {
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().triangularView<Eigen::Lower>());
  }
};

template <> struct GemmT<Dense, Dense> {
  static void fn(MatrixRef dst, MatrixRef const &lhs, MatrixRef const &rhs,
                 Scalar alpha) {
    dst.noalias() += alpha * lhs * rhs.transpose();
  }
};
