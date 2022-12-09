#include "proxnlp/math.hpp"
#include <type_traits>

namespace proxnlp {

namespace block_chol {

namespace backend {

using Scalar = double;
using Stride = Eigen::OuterStride<Eigen::Dynamic>;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixRef = Eigen::Ref<Matrix>;
using MatrixMap = Eigen::Map<Matrix, Eigen::Unaligned, Stride>;

using usize = decltype(sizeof(0));
using isize = typename std::make_signed<usize>::type;
static constexpr usize ALIGN = 64;
static constexpr isize RECURSION_THRESHOLD = 16;

enum BlockKind {
  Zero,
  Diag,
  TriL,
  TriU,
  Dense,
};

template <BlockKind LHS, BlockKind RHS> struct GemmT;
template <BlockKind LHS> struct GemmTLower;

// cases where one of the operands is zero
template <BlockKind RHS> struct GemmT<Zero, RHS> {
  static void fn(MatrixRef /*lhs*/, MatrixRef /*rhs*/, Scalar /*alpha*/) {}
};
template <BlockKind LHS> struct GemmT<LHS, Zero> {
  static void fn(MatrixRef /*dst*/, MatrixRef /*lhs*/, MatrixRef /*rhs*/,
                 Scalar /*alpha*/) {}
};
template <> struct GemmT<Zero, Zero> {
  static void fn(MatrixRef /*dst*/, MatrixRef /*lhs*/, MatrixRef /*rhs*/,
                 Scalar /*alpha*/) {}
};
template <> struct GemmTLower<Zero> {
  static void fn(MatrixRef /*dst*/, MatrixRef /*lhs*/, MatrixRef /*rhs*/,
                 Scalar /*alpha*/) {}
};

// LHS is Diag
template <> struct GemmTLower<Diag> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.diagonal() += alpha * lhs.diagonal().cwiseProduct(rhs.diagonal());
  }
};
template <> struct GemmT<Diag, Diag> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    isize n = dst.rows();
    for (isize j = 0; j < n; ++j) {
      dst(j, j) += alpha * lhs(j, j) * rhs(j, j);
    }
  }
};

template <> struct GemmT<Diag, TriL> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
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
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
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
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst += alpha * (lhs.diagonal().asDiagonal() * rhs.transpose());
  }
};

// LHS is TriL
template <> struct GemmT<TriL, Diag> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
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
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    isize n = lhs.rows();

    if (n <= RECURSION_THRESHOLD) {
      alignas(ALIGN) Scalar lhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];
      alignas(ALIGN) Scalar rhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];

      MatrixMap lhs_map{lhs_copy, n, n, Stride(RECURSION_THRESHOLD)};
      MatrixMap rhs_map{rhs_copy, n, n, Stride(RECURSION_THRESHOLD)};

      lhs_map.triangularView<Eigen::Lower>() = lhs;
      lhs_map.triangularView<Eigen::StrictlyUpper>().setZero();
      rhs_map.triangularView<Eigen::Lower>() = rhs;
      rhs_map.triangularView<Eigen::StrictlyUpper>().setZero();

      dst.noalias() += alpha * (lhs_map * rhs_map.transpose());

      return;
    }

    isize bs = n / 2;
    isize rem = n - bs;

    //     [A00    ]
    // A = [A10 A11]
    //     [B00.T B10.T]
    // B.T=[      B11.T]

    //         [A00×B00.T A00×B10.T            ]
    // A×B.T = [A10×B00.T A10×B10.T + A11×B11.T]

    MatrixRef lhs00 = lhs.topLeftCorner(bs, bs);
    MatrixRef lhs10 = lhs.bottomLeftCorner(rem, bs);
    MatrixRef lhs11 = lhs.bottomRightCorner(rem, rem);

    MatrixRef rhs00 = rhs.topLeftCorner(bs, bs);
    MatrixRef rhs10 = rhs.bottomLeftCorner(rem, bs);
    MatrixRef rhs11 = rhs.bottomRightCorner(rem, rem);

    MatrixRef dst00 = dst.topLeftCorner(bs, bs);
    MatrixRef dst01 = dst.topRightCorner(bs, rem);
    MatrixRef dst10 = dst.bottomLeftCorner(rem, bs);
    MatrixRef dst11 = dst.bottomRightCorner(rem, rem);

    fn(dst00, lhs00, rhs00, alpha);
    dst01.noalias() +=
        alpha * (lhs00.triangularView<Eigen::Lower>() * rhs10.transpose());
    dst10.noalias() +=
        alpha * (lhs10 * rhs00.transpose().triangularView<Eigen::Upper>());
    dst11.noalias() += alpha * (lhs10 * rhs10.transpose());
    fn(dst11, lhs11, rhs11, alpha);
  }
};

template <> struct GemmTLower<TriL> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    isize n = lhs.rows();

    if (n <= RECURSION_THRESHOLD) {
      alignas(ALIGN) Scalar lhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];
      alignas(ALIGN) Scalar rhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];

      MatrixMap lhs_map{lhs_copy, n, n, Stride(RECURSION_THRESHOLD)};
      MatrixMap rhs_map{rhs_copy, n, n, Stride(RECURSION_THRESHOLD)};

      lhs_map.triangularView<Eigen::Lower>() = lhs;
      lhs_map.triangularView<Eigen::StrictlyUpper>().setZero();
      rhs_map.triangularView<Eigen::Lower>() = rhs;
      rhs_map.triangularView<Eigen::StrictlyUpper>().setZero();

      dst.triangularView<Eigen::Lower>() += lhs_map * rhs_map.transpose();

      return;
    }

    isize bs = n / 2;
    isize rem = n - bs;

    //     [A00    ]
    // A = [A10 A11]
    //     [B00.T B10.T]
    // B.T=[      B11.T]

    //         [A00×B00.T A00×B10.T            ]
    // A×B.T = [A10×B00.T A10×B10.T + A11×B11.T]

    MatrixRef lhs00 = lhs.topLeftCorner(bs, bs);
    MatrixRef lhs10 = lhs.bottomLeftCorner(rem, bs);
    MatrixRef lhs11 = lhs.bottomRightCorner(rem, rem);

    MatrixRef rhs00 = rhs.topLeftCorner(bs, bs);
    MatrixRef rhs10 = rhs.bottomLeftCorner(rem, bs);
    MatrixRef rhs11 = rhs.bottomRightCorner(rem, rem);

    MatrixRef dst00 = dst.topLeftCorner(bs, bs);
    MatrixRef dst10 = dst.bottomLeftCorner(rem, bs);
    MatrixRef dst11 = dst.bottomRightCorner(rem, rem);

    fn(dst00, lhs00, rhs00, alpha);
    dst10.noalias() +=
        alpha * (lhs10 * rhs00.transpose().triangularView<Eigen::Upper>());
    dst11.noalias() += alpha * (lhs10 * rhs10.transpose());
    fn(dst11, lhs11, rhs11, alpha);
  }
};

template <> struct GemmT<TriL, TriU> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    isize n = lhs.rows();

    if (n <= RECURSION_THRESHOLD) {
      alignas(ALIGN) Scalar lhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];
      alignas(ALIGN) Scalar rhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];

      MatrixMap lhs_map{lhs_copy, n, n, Stride(RECURSION_THRESHOLD)};
      MatrixMap rhs_map{rhs_copy, n, n, Stride(RECURSION_THRESHOLD)};

      lhs_map.triangularView<Eigen::Lower>() = lhs;
      lhs_map.triangularView<Eigen::StrictlyUpper>().setZero();
      rhs_map.triangularView<Eigen::Upper>() = rhs;
      rhs_map.triangularView<Eigen::StrictlyLower>().setZero();

      dst.noalias() += alpha * (lhs_map * rhs_map.transpose());

      return;
    }

    isize bs = n / 2;
    isize rem = n - bs;

    //     [A00    ]
    // A = [A10 A11]
    //     [B00.T      ]
    // B.T=[B01.T B11.T]

    //         [A00×B00.T
    // A×B.T = [A10×B00.T + A11×B01.T  A11×B11.T]

    MatrixRef lhs00 = lhs.topLeftCorner(bs, bs);
    MatrixRef lhs10 = lhs.bottomLeftCorner(rem, bs);
    MatrixRef lhs11 = lhs.bottomRightCorner(rem, rem);

    MatrixRef rhs00 = rhs.topLeftCorner(bs, bs);
    MatrixRef rhs01 = rhs.topRightCorner(bs, rem);
    MatrixRef rhs11 = rhs.bottomRightCorner(rem, rem);

    MatrixRef dst00 = dst.topLeftCorner(bs, bs);
    MatrixRef dst10 = dst.bottomLeftCorner(rem, bs);
    MatrixRef dst11 = dst.bottomRightCorner(rem, rem);

    fn(dst00, lhs00, rhs00, alpha);
    dst10.noalias() +=
        alpha * (lhs10 * rhs00.transpose().triangularView<Eigen::Lower>());
    dst10.noalias() +=
        alpha * (lhs11.triangularView<Eigen::Lower>() * rhs01.transpose());
    fn(dst11, lhs11, rhs11, alpha);
  }
};

template <> struct GemmT<TriL, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() +=
        lhs.template triangularView<Eigen::Lower>() * (alpha * rhs.transpose());
  }
};

// LHS is TriU
template <> struct GemmT<TriU, Diag> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
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
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    isize n = lhs.rows();

    if (n <= RECURSION_THRESHOLD) {
      alignas(ALIGN) Scalar lhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];
      alignas(ALIGN) Scalar rhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];

      MatrixMap lhs_map{lhs_copy, n, n, Stride(RECURSION_THRESHOLD)};
      MatrixMap rhs_map{rhs_copy, n, n, Stride(RECURSION_THRESHOLD)};

      lhs_map.triangularView<Eigen::Upper>() = lhs;
      lhs_map.triangularView<Eigen::StrictlyLower>().setZero();
      rhs_map.triangularView<Eigen::Lower>() = rhs;
      rhs_map.triangularView<Eigen::StrictlyUpper>().setZero();

      dst.noalias() += alpha * (lhs_map * rhs_map.transpose());

      return;
    }

    isize bs = n / 2;
    isize rem = n - bs;

    //     [A00 A01]
    // A = [    A11]
    //     [B00.T B10.T]
    // B.T=[      B11.T]

    //         [A00×B00.T  A00×B10.T + A01×B11.T]
    // A×B.T = [           A11×B11.T            ]

    MatrixRef lhs00 = lhs.topLeftCorner(bs, bs);
    MatrixRef lhs01 = lhs.topRightCorner(bs, rem);
    MatrixRef lhs11 = lhs.bottomRightCorner(rem, rem);

    MatrixRef rhs00 = rhs.topLeftCorner(bs, bs);
    MatrixRef rhs10 = rhs.bottomLeftCorner(rem, bs);
    MatrixRef rhs11 = rhs.bottomRightCorner(rem, rem);

    MatrixRef dst00 = dst.topLeftCorner(bs, bs);
    MatrixRef dst01 = dst.topRightCorner(bs, rem);
    MatrixRef dst11 = dst.bottomRightCorner(rem, rem);

    fn(dst00, lhs00, rhs00, alpha);
    dst01.noalias() +=
        alpha * (lhs00.triangularView<Eigen::Upper>() * rhs10.transpose());
    dst01.noalias() +=
        alpha * (lhs01 * rhs11.transpose().triangularView<Eigen::Upper>());
    fn(dst11, lhs11, rhs11, alpha);
  }
};

template <> struct GemmT<TriU, TriU> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    isize n = lhs.rows();

    if (n <= RECURSION_THRESHOLD) {
      alignas(ALIGN) Scalar lhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];
      alignas(ALIGN) Scalar rhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];

      MatrixMap lhs_map{lhs_copy, n, n, Stride(RECURSION_THRESHOLD)};
      MatrixMap rhs_map{rhs_copy, n, n, Stride(RECURSION_THRESHOLD)};

      lhs_map.triangularView<Eigen::Upper>() = lhs;
      lhs_map.triangularView<Eigen::StrictlyLower>().setZero();
      rhs_map.triangularView<Eigen::Upper>() = rhs;
      rhs_map.triangularView<Eigen::StrictlyLower>().setZero();

      dst.noalias() += alpha * (lhs_map * rhs_map.transpose());

      return;
    }

    isize bs = n / 2;
    isize rem = n - bs;

    //     [A00 A01]
    // A = [    A11]
    //     [B00.T      ]
    // B.T=[B01.T B11.T]

    //         [A00×B00.T + A01×B01.T  A01×B11.T]
    // A×B.T = [A11×B01.T              A11×B11.T]

    MatrixRef lhs00 = lhs.topLeftCorner(bs, bs);
    MatrixRef lhs01 = lhs.topRightCorner(bs, rem);
    MatrixRef lhs11 = lhs.bottomRightCorner(rem, rem);

    MatrixRef rhs00 = rhs.topLeftCorner(bs, bs);
    MatrixRef rhs01 = rhs.topRightCorner(bs, rem);
    MatrixRef rhs11 = rhs.bottomRightCorner(rem, rem);

    MatrixRef dst00 = dst.topLeftCorner(bs, bs);
    MatrixRef dst01 = dst.topRightCorner(bs, rem);
    MatrixRef dst10 = dst.bottomLeftCorner(rem, bs);
    MatrixRef dst11 = dst.bottomRightCorner(rem, rem);

    fn(dst00, lhs00, rhs00, alpha);
    dst00.noalias() += alpha * (lhs01 * rhs01.transpose());
    dst01.noalias() +=
        alpha * (lhs01 * rhs11.transpose().triangularView<Eigen::Lower>());
    dst10.noalias() +=
        alpha * (lhs11.triangularView<Eigen::Upper>() * rhs01.transpose());
    fn(dst11, lhs11, rhs11, alpha);
  }
};

template <> struct GemmTLower<TriU> {
  // dst is triu
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    isize n = lhs.rows();

    if (n <= RECURSION_THRESHOLD) {
      alignas(ALIGN) Scalar lhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];
      alignas(ALIGN) Scalar rhs_copy[RECURSION_THRESHOLD * RECURSION_THRESHOLD];

      MatrixMap lhs_map{lhs_copy, n, n, Stride(RECURSION_THRESHOLD)};
      MatrixMap rhs_map{rhs_copy, n, n, Stride(RECURSION_THRESHOLD)};

      lhs_map.triangularView<Eigen::Upper>() = lhs;
      lhs_map.triangularView<Eigen::StrictlyLower>().setZero();
      rhs_map.triangularView<Eigen::Upper>() = rhs;
      rhs_map.triangularView<Eigen::StrictlyLower>().setZero();

      dst.triangularView<Eigen::Lower>() +=
          alpha * (lhs_map * rhs_map.transpose());

      return;
    }

    isize bs = n / 2;
    isize rem = n - bs;

    //     [A00 A01]
    // A = [    A11]
    //     [B00.T      ]
    // B.T=[B01.T B11.T]

    //         [A00×B00.T + A01×B01.T  A01×B11.T]
    // A×B.T = [A11×B01.T              A11×B11.T]

    MatrixRef lhs00 = lhs.topLeftCorner(bs, bs);
    MatrixRef lhs01 = lhs.topRightCorner(bs, rem);
    MatrixRef lhs11 = lhs.bottomRightCorner(rem, rem);

    MatrixRef rhs00 = rhs.topLeftCorner(bs, bs);
    MatrixRef rhs01 = rhs.topRightCorner(bs, rem);
    MatrixRef rhs11 = rhs.bottomRightCorner(rem, rem);

    MatrixRef dst00 = dst.topLeftCorner(bs, bs);
    MatrixRef dst10 = dst.bottomLeftCorner(rem, bs);
    MatrixRef dst11 = dst.bottomRightCorner(rem, rem);

    fn(dst00, lhs00, rhs00, alpha);
    dst00.noalias() += alpha * (lhs01 * rhs01.transpose());
    dst10.noalias() +=
        alpha * (lhs11.triangularView<Eigen::Upper>() * rhs01.transpose());
    fn(dst11, lhs11, rhs11, alpha);
  }
};

template <> struct GemmT<TriU, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs.template triangularView<Eigen::Upper>() * rhs.transpose());
  }
};

// LHS is Dense
template <> struct GemmT<Dense, Diag> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() += alpha * (lhs * rhs.transpose().diagonal().asDiagonal());
  }
};

template <> struct GemmT<Dense, TriL> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().triangularView<Eigen::Upper>());
  }
};

template <> struct GemmT<Dense, TriU> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().triangularView<Eigen::Lower>());
  }
};

template <> struct GemmTLower<Dense> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.triangularView<Eigen::Lower>() += alpha * (lhs * rhs.transpose());
  }
};
template <> struct GemmT<Dense, Dense> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() += alpha * (lhs * rhs.transpose());
  }
};

} // namespace backend
} // namespace block_chol
} // namespace proxnlp
