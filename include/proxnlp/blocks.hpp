/// @file
/// @author Sarah El-Kazdadi
/// @brief Routines for block-sparse (notably, KKT-type) matrix LDLT
/// factorisation.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/math.hpp"
#include <Eigen/Cholesky>
#include <type_traits>

#include <algorithm>
#include <numeric>
#include "linalg/block-kind.hpp"

#include <iostream>

namespace proxnlp {
/// @brief	Block-wise Cholesky or LDLT factorisation routines.
namespace block_chol {

using Scalar = double;
using MatrixRef = math_types<Scalar>::MatrixRef;
using ConstMatrixRef = math_types<Scalar>::ConstMatrixRef;
using VectorRef = math_types<Scalar>::VectorRef;

using isize = std::int64_t;

struct SymbolicBlockMatrix {
  BlockKind *data;
  isize *segment_lens;
  isize segments_count;
  isize outer_stride;

  isize nsegments() const noexcept { return segments_count; }
  isize size() const noexcept { return segments_count * outer_stride; }
  BlockKind *ptr(isize i, isize j) const noexcept {
    return data + (i + j * outer_stride);
  }

  /// Get the symbolic submatrix starting from the block in position (i, i).
  SymbolicBlockMatrix submatrix(isize i, isize n) const noexcept;
  BlockKind &operator()(isize i, isize j) const noexcept { return *ptr(i, j); }

  /// Deep copy of the struct, possibily with a permutation.
  void deep_copy(SymbolicBlockMatrix const &in,
                 isize const *perm = nullptr) const noexcept;

  /// Brute-force search of the best permutation possible in the block matrix.
  /// work has length `in.nsegments()`
  Eigen::ComputationInfo
  brute_force_best_permutation(SymbolicBlockMatrix const &in, isize *best_perm,
                               isize *iwork) const;
  isize count_nnz() const noexcept;
  /// Perform symbolic block-wise LLT decomposition;
  /// the output sparsity pattern should be that of the matrix \f$L\f$
  /// of the Cholesky decomposition.
  bool llt_in_place() const noexcept;
};

void print_sparsity_pattern(const SymbolicBlockMatrix &smat) noexcept;





namespace backend {

/// At the end of the execution, \param a contains
/// the lower-triangular matrix \f$L\f$ in the LDLT decomposition.
/// More precisely: a stores L -sans its diagonal which is all ones.
/// The diagonal of \param a contains the diagonal matrix \f$D\f$.
inline void ldlt_in_place_unblocked(MatrixRef a) {
  isize n = a.rows();
  assert(n == diag.size());
  if (n == 0) {
    return;
  }

  isize j = 0;
  while (true) {
    auto l10 = a.row(j).head(j);
    auto d0 = a.diagonal().head(j);
    auto work = a.col(n - 1).head(j);

    work = l10.transpose().cwiseProduct(d0);
    a(j, j) -= work.dot(l10);

    if (j + 1 == n) {
      return;
    }

    isize rem = n - j - 1;

    auto l20 = a.bottomLeftCorner(rem, j);
    auto l21 = a.col(j).tail(rem);

    l21.noalias() -= l20 * work;
    l21 *= 1 / a(j, j);
    ++j;
  }
}

inline void ldlt_in_place_recursive(MatrixRef const &a) {
  isize n = a.rows();
  if (n <= 128) {
    backend::ldlt_in_place_unblocked(a);
  } else {
    isize bs = (n + 1) / 2;
    isize rem = n - bs;

    auto a_mut = a.const_cast_derived();
    MatrixRef l00 = a_mut.block(0, 0, bs, bs);
    MatrixRef l10 = a_mut.block(bs, 0, rem, bs);
    MatrixRef l11 = a_mut.block(bs, bs, rem, rem);

    backend::ldlt_in_place_recursive(l00);
    auto d0 = l00.diagonal();

    l00.transpose()
        .template triangularView<Eigen::UnitUpper>()
        .template solveInPlace<Eigen::OnTheRight>(l10);

    MatrixRef work = a_mut.block(0, rem, rem, bs);
    work = l10;
    l10 = l10 * d0.asDiagonal().inverse();

    l11.template triangularView<Eigen::Lower>() -= l10 * work.transpose();

    backend::ldlt_in_place_recursive(l11);
  }
}

/// A recursive, in-place implementation of the LDLT decomposition.
/// To be applied to dense blocks.
inline void dense_ldlt_in_place(MatrixRef a) { ldlt_in_place_recursive(a); }

using Eigen::internal::LDLT_Traits;

/// Taking the decomposed LDLT matrix \param mat, solve the original linear
/// system.
inline bool dense_ldlt_solve_in_place(ConstMatrixRef const &mat, MatrixRef b) {
  typedef LDLT_Traits<ConstMatrixRef, Eigen::Lower> Traits;
  Traits::getL(mat).solveInPlace(b);

  using std::abs;
  Eigen::Diagonal<const ConstMatrixRef>::RealReturnType vecD(mat.diagonal());
  const Scalar tol = std::numeric_limits<Scalar>::min();
  for (isize i = 0; i < vecD.size(); ++i) {
    if (abs(vecD(i)) > tol)
      b.row(i) /= vecD(i);
    else
      b.row(i).setZero();
  }

  Traits::getU(mat).solveInPlace(b);
  return true;
}

inline MatrixRef::PlainMatrix
dense_ldlt_reconstruct(ConstMatrixRef const &mat) {
  typedef LDLT_Traits<ConstMatrixRef, Eigen::Lower> Traits;
  MatrixRef::PlainMatrix res(mat.rows(), mat.cols());

  res.setIdentity();
  res = Traits::getU(mat) * res;

  auto vecD = mat.diagonal();
  res = vecD.asDiagonal() * res;

  res = Traits::getL(mat) * res;
  return res;
}

#if true

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
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.diagonal() +=
        alpha * lhs.diagonal().cwiseProduct(rhs.transpose().diagonal());
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
  // dst is tril
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
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template triangularView<Eigen::Upper>());
    dst.noalias() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Upper>());
  }
};

template <> struct GemmT<TriL, TriU> {
  // dst is tril
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    // PERF
    // dst += alpha * (lhs.template triangularView<Eigen::Lower>() *
    //                 rhs.transpose().template triangularView<Eigen::Lower>());
    dst.triangularView<Eigen::Lower>() +=
        alpha * (lhs * rhs.transpose().template triangularView<Eigen::Lower>());
  }
};

template <> struct GemmT<TriL, Dense> {
  // dst is dense
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() +=
        lhs.template triangularView<Eigen::Lower>() * (alpha * rhs.transpose());
  }
};

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
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
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
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() += (lhs.template triangularView<Eigen::Upper>() *
                      (alpha * rhs.transpose()));
  }
};

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

template <> struct GemmT<Dense, Dense> {
  static void fn(MatrixRef dst, MatrixRef lhs, MatrixRef rhs, Scalar alpha) {
    dst.noalias() += alpha * lhs * rhs.transpose();
  }
};
#else

#include "linalg/gemmt.hpp"

#endif

inline void gemmt(MatrixRef const &dst, MatrixRef const &lhs,
                  MatrixRef const &rhs, BlockKind lhs_kind, BlockKind rhs_kind,
                  Scalar alpha) {
  // dst += alpha * lhs * rhs.Scalar
  switch (lhs_kind) {
  case Zero: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Zero, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Zero, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Zero, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Zero, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Zero, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case Diag: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Diag, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Diag, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Diag, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Diag, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Diag, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case TriL: {
    switch (rhs_kind) {
    case Zero:
      GemmT<TriL, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<TriL, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<TriL, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<TriL, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<TriL, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case TriU: {
    switch (rhs_kind) {
    case Zero:
      GemmT<TriU, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<TriU, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<TriU, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<TriU, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<TriU, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  case Dense: {
    switch (rhs_kind) {
    case Zero:
      GemmT<Dense, Zero>::fn(dst, lhs, rhs, alpha);
      break;
    case Diag:
      GemmT<Dense, Diag>::fn(dst, lhs, rhs, alpha);
      break;
    case TriL:
      GemmT<Dense, TriL>::fn(dst, lhs, rhs, alpha);
      break;
    case TriU:
      GemmT<Dense, TriU>::fn(dst, lhs, rhs, alpha);
      break;
    case Dense:
      GemmT<Dense, Dense>::fn(dst, lhs, rhs, alpha);
      break;
    }
    break;
  }
  }
}
} // namespace backend

struct DenseLDLT {
  using MatrixXs = MatrixRef::PlainMatrix;
  MatrixXs m_matrix;
  bool permutate = false;
  DenseLDLT() = default;
  explicit DenseLDLT(isize n) : m_matrix(n, n) { m_matrix.setZero(); }
  explicit DenseLDLT(MatrixRef a) : m_matrix(a) {
    backend::dense_ldlt_in_place(m_matrix);
  }

  DenseLDLT &compute(MatrixRef a) {
    m_matrix = a;
    backend::dense_ldlt_in_place(m_matrix);
    return *this;
  }

  void solveInPlace(MatrixRef b) const {
    backend::dense_ldlt_solve_in_place(m_matrix, b);
  }

  MatrixXs reconstructedMatrix() const {
    return backend::dense_ldlt_reconstruct(m_matrix);
  }

  Eigen::Diagonal<const MatrixXs> vectorD() const {
    return m_matrix.diagonal();
  }
};

/// @brief Block matrix data structure with LDLT algos.
struct BlockLDLT {
  using Traits = backend::LDLT_Traits<MatrixRef, Eigen::Lower>;
  using MatrixXs = MatrixRef::PlainMatrix;

  MatrixRef m_matrix;
  SymbolicBlockMatrix m_structure;
  Eigen::ComputationInfo m_info;

  // BlockLDLT() = default;
  // explicit BlockLDLT(isize size)
  //   : storage(size, size), structure()
  // {}

  BlockLDLT(MatrixRef mat, SymbolicBlockMatrix structure)
      : m_matrix(mat), m_structure(structure) {}

  void setStructure(SymbolicBlockMatrix structure) { m_structure = structure; }

  Eigen::ComputationInfo info() const { return m_info; }

  MatrixXs reconstructedMatrix() const {
    return backend::dense_ldlt_reconstruct(m_matrix);
  }

  /// TODO: make block-sparse variant of solveInPlace()
  bool solveInPlace(MatrixRef b) const {
    return backend::dense_ldlt_solve_in_place(m_matrix, b);
  }

  void permute(BlockLDLT in, isize const *perm) {
    MatrixRef mat(m_matrix);

    m_structure.deep_copy(in.m_structure, perm);

    isize nblocks = in.m_structure.nsegments();

    isize out_offset_i = 0;
    for (isize i = 0; i < nblocks; ++i) {
      isize bsi = m_structure.segment_lens[i];

      isize in_offset_i = 0;
      for (isize ii = 0; ii < perm[i]; ++ii) {
        in_offset_i += in.m_structure.segment_lens[ii];
      }

      isize out_offset_j = 0;
      for (isize j = 0; j < nblocks; ++j) {
        isize bsj = m_structure.segment_lens[j];

        isize in_offset_j = 0;
        for (isize jj = 0; jj < perm[j]; ++jj) {
          in_offset_j += in.m_structure.segment_lens[jj];
        }

        for (isize i_inner = 0; i_inner < bsi; ++i_inner) {
          for (isize j_inner = 0; j_inner < bsj; ++j_inner) {
            mat(out_offset_i + i_inner, out_offset_j + j_inner) =
                in.m_matrix(in_offset_i + i_inner, in_offset_j + j_inner);
          }
        }

        out_offset_j += bsj;
      }

      out_offset_i += bsi;
    }
  }

  bool ldlt_in_place_impl() {

    isize nblocks = m_structure.nsegments();
    isize n = m_matrix.rows();
    if (nblocks == 0) {
      return true;
    }

    BlockKind &structure_00 = m_structure(0, 0);
    isize bs = m_structure.segment_lens[0];
    isize rem = n - bs;
    MatrixRef store_mut = m_matrix.const_cast_derived();
    MatrixRef l00 = store_mut.block(0, 0, bs, bs);
    MatrixRef l11 = store_mut.block(bs, bs, rem, rem);
    auto d0 = l00.diagonal();

    MatrixRef work = store_mut.block(0, rem, rem, bs);

    switch (structure_00) {
    case Zero:
    case TriU:
    case Dense:
      return false;

    case TriL: {
      // compute l00
      backend::dense_ldlt_in_place(l00);

      isize offset = bs;

      for (isize i = 1; i < nblocks; ++i) {
        isize bsi = m_structure.segment_lens[i];
        MatrixRef li0 = store_mut.block(offset, 0, bsi, bs);
        MatrixRef li0_copy = work.block(offset - bs, 0, bsi, bs);

        switch (m_structure(i, 0)) {
        case Diag:
        case TriL:
          return false;
        case TriU: {
          auto li0_u = li0.template triangularView<Eigen::Upper>();
          // PERF: replace li0.Scalar by li0_tl
          // auto li0_tl = li0.transpose().template
          // triangularView<Eigen::Lower>();
          l00.template triangularView<Eigen::UnitLower>().solveInPlace(
              li0.transpose());
          li0_copy.template triangularView<Eigen::Upper>() = li0_u;
          li0_u = li0 * d0.asDiagonal().inverse();
          break;
        }
        case Dense: {
          l00.template triangularView<Eigen::UnitLower>().solveInPlace(
              li0.transpose());
          li0_copy = li0;
          li0 = li0 * d0.asDiagonal().inverse();
          break;
        }
        case Zero:
          break;
        }
      }

      break;
    }
    case Diag: {
      // l00 is unchanged
      isize offset = bs;

      for (isize i = 1; i < nblocks; ++i) {
        isize bsi = m_structure.segment_lens[i];
        MatrixRef li0 = store_mut.block(offset, 0, bsi, bs);
        MatrixRef li0_copy = work.block(offset - bs, 0, bsi, bs);

        switch (m_structure(i, 0)) {
        case TriL:
          return false;
        case Diag: {
          auto li0_d = li0.diagonal();
          li0_copy.diagonal() = li0_d;
          li0_d = li0_d.cwiseQuotient(d0);
          break;
        }
        case TriU: {
          auto li0_u = li0.template triangularView<Eigen::Upper>();
          li0_copy.template triangularView<Eigen::Upper>() = li0_u;
          li0_u = li0 * d0.asDiagonal().inverse();
          break;
        }
        case Dense: {
          li0_copy = li0;
          li0 = li0 * d0.asDiagonal().inverse();
          break;
        }
        case Zero:
          break;
        }
        offset += bsi;
      }

      break;
    }
    }

    isize offset_i = bs;
    for (isize i = 1; i < nblocks; ++i) {
      isize bsi = m_structure.segment_lens[i];
      MatrixRef li0 = store_mut.block(offset_i, 0, bsi, bs);
      MatrixRef li0_prev = work.block(offset_i - bs, 0, bsi, bs);

      MatrixRef target_ii = store_mut.block(offset_i, offset_i, bsi, bsi);

      // target_ii -= li0 * li0_prev.Scalar;
      backend::gemmt(target_ii, li0, li0_prev, m_structure(i, 0),
                     m_structure(i, 0), Scalar(-1));

      isize offset_j = offset_i + bsi;
      for (isize j = i + 1; j < nblocks; ++j) {
        // target_ji -= lj0 * li0_prev.Scalar

        isize bsj = m_structure.segment_lens[j];
        MatrixRef lj0 = store_mut.block(offset_j, 0, bsj, bs);
        MatrixRef target_ji = store_mut.block(offset_j, offset_i, bsj, bsi);

        backend::gemmt(target_ji, lj0, li0_prev, m_structure(j, 0),
                       m_structure(i, 0), Scalar(-1));

        offset_j += bsj;
      }

      offset_i += bsi;
    }

    return BlockLDLT{
        l11,
        m_structure.submatrix(1, nblocks - 1),
    }
        .ldlt_in_place_impl();
  }

  BlockLDLT &compute(MatrixRef mat) {
    m_matrix = mat;
    m_info = ldlt_in_place_impl() ? Eigen::Success : Eigen::NumericalIssue;

    return *this;
  }
};

} // namespace block_chol
} // namespace proxnlp
