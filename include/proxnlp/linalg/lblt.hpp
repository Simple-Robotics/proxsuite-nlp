#pragma once

#include "proxnlp/linalg/block-diag.hpp"
#include "proxnlp/linalg/dense.hpp"

namespace proxnlp {
namespace linalg {

template <typename MatrixType> struct LBLT_Traits {
  using Scalar = typename MatrixType::Scalar;
  using MatrixL = Eigen::TriangularView<MatrixType, Eigen::UnitLower>;
  using MatrixU =
      Eigen::TriangularView<const typename MatrixType::TransposeReturnType,
                            Eigen::UnitLower>;
  using MatrixB = BlockDiagonalMatrix<Scalar, 2, MatrixType::Options>;
  static auto getL(const MatrixType &m) { return MatrixL(m); }
  static auto getU(const MatrixType &m) { return MatrixU(m.transpose()); }
};

/// The Bunch-Kaufman algorithm. Factorizes a symmetric (indefinite) matrix
/// @f$A@f$ into a triangular and block-diagonal matrix product @f$ LBL^\top
/// @f$.
template <typename _MatrixType> struct BunchKaufman {
public:
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using Matrix2s = Eigen::Matrix<Scalar, 2, 2, Eigen::ColMajor>;
  using Traits = LBLT_Traits<MatrixType>;
  using MatrixB = typename Traits::MatrixB;

  template <typename Derived>
  bool solveInPlace(Eigen::MatrixBase<Derived> &bAndX) const;

  BunchKaufman &compute(const MatrixType &matrix);

private:
  MatrixType m_matrix;
  MatrixB m_blck;
};

} // namespace linalg
} // namespace proxnlp
