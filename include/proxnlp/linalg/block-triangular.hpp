/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./blocks.hpp"

namespace proxnlp {
namespace linalg {
namespace backend {

template <int Mode, bool IsLower = (Mode & Eigen::Lower) == Eigen::Lower>
struct block_triangular_subsolve_impl;

} // namespace backend

/// @brief    Representation for triangular block matrices.
/// @details  Provides a convenience function for solving linear systems
/// in-place.
template <typename _MatrixType, int _Mode> struct TriangularBlockMatrix {
public:
  enum { Mode = _Mode, IsLower = (Mode & Eigen::Lower) == Eigen::Lower };
  using MatrixType = _MatrixType;
  using MatrixTypeNested =
      typename Eigen::internal::ref_selector<MatrixType>::non_const_type;
  using Scalar = typename MatrixType::Scalar;

  TriangularBlockMatrix(MatrixType &mat, const SymbolicBlockMatrix &structure)
      : m_matrix(mat), m_structure(structure) {}

  /// @brief  Block-sparse variant of the TriangularViewType::solveInPlace()
  /// method on standard dense matrices.
  template <typename Derived, bool UseBlockGemmT = true>
  bool solveInPlace(Eigen::MatrixBase<Derived> &bAndX) const {

    assert(bAndX.rows() == m_matrix.cols());

    const isize size = bAndX.rows();
    isize rem = size;
    const isize nblocks = m_structure.nsegments();

    // loop over structure rows
    for (isize i = IsLower ? 0 : nblocks - 1; IsLower ? i < nblocks : i >= 0;
         IsLower ? i += 1 : i -= 1) {

      isize n0 = m_structure.segment_lens[i];

      // current window size is rem_i
      // in lower mode: it is bottom right corner of m_matrix
      // in upper mode: top left corner
      auto L_cur = IsLower ? m_matrix.bottomRightCorner(rem, rem)
                           : m_matrix.topLeftCorner(rem, rem);
      auto b_cur = IsLower ? bAndX.bottomRows(rem) // look down
                           : bAndX.topRows(rem);   // look up

      // next subproblem size is rem_i+1 = rem_i - n0
      rem -= n0;

      auto b0 = IsLower ? b_cur.topRows(n0) : b_cur.bottomRows(n0);
      auto b1 = IsLower ? b_cur.bottomRows(rem) : b_cur.topRows(rem);

      auto L00 = IsLower ? L_cur.topLeftCorner(n0, n0)
                         : L_cur.bottomRightCorner(n0, n0);

      auto L10 = IsLower ? L_cur.bottomLeftCorner(rem, n0)
                         : L_cur.topRightCorner(rem, n0);

      // step 1: solve b0
      bool inner_flag = backend::block_triangular_subsolve_impl<Mode>::run(
          L00, b0, m_structure(i, i));
      if (!inner_flag)
        return false;

      // step 2: reformulate the problem for the following rows

      if (!UseBlockGemmT) {
        b1.noalias() -= L10 * b0;
      } else {

        // perform the multiplication in a block aware manner
        // in Lower mode: move down from block (i, i) until (i, nb-1)
        // in Upper mode: move down from block (0, nb-1) until (i, nb-1)
        isize p0 = 0;
        for (isize p = IsLower ? i + 1 : 0; IsLower ? p < nblocks : p < i;
             ++p) {
          isize n_c = m_structure.segment_lens[p];
          auto L10_blk = L10.middleRows(p0, n_c); // size (n_c, n0)
          auto dst = b1.middleRows(p0, n_c);      // size (n_c)
          // b0 has size n0
          // take the block out of L10
          BlockKind lhs_kind = m_structure(p, i);
          backend::gemmt<Scalar>::run(dst, L10_blk, b0.transpose(), lhs_kind,
                                      Dense, Scalar(-1));
          p0 += n_c;
        }
      }
    }

    assert(rem == 0);

    return true;
  }

protected:
  MatrixTypeNested m_matrix;
  SymbolicBlockMatrix m_structure; // shallow copy of ctor input
};

namespace backend {

template <int Mode, bool IsLower> struct block_triangular_subsolve_impl {
  static constexpr bool HasUnitDiag =
      (Mode & Eigen::UnitDiag) == Eigen::UnitDiag;
  static constexpr BlockKind WhichTriValid = IsLower ? TriL : TriU;

  template <typename MatType, typename OutType>
  static bool run(const MatType &L00, OutType &b0, BlockKind kind) {
    switch (kind) {
    case Zero:
    case Dense:
      return false;
    case Diag: {
      if (!HasUnitDiag)
        b0 = L00.diagonal().asDiagonal().inverse() * b0;
      break;
    }
    case WhichTriValid: {
      // get triangular view of block, solve for b0
      // the chosen mode in the template propagates to the diagonal blocks
      L00.template triangularView<Mode>().solveInPlace(b0);
      break;
    }
    default:
      break;
    }
    return true;
  }
};

} // namespace backend

} // namespace linalg
} // namespace proxnlp
