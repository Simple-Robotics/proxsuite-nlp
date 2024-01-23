/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @brief Routines for block-sparse matrix LDLT factorisation.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/linalg/dense.hpp"
#include "proxsuite-nlp/linalg/block-triangular.hpp"

#include "proxsuite-nlp/linalg/gemmt.hpp"

#include <numeric>

namespace proxsuite {
namespace nlp {
namespace linalg {

namespace backend {

/// Implementation struct for the recursive block LDLT algorithm.
template <typename Scalar> struct block_impl {
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  MatrixRef mat;
  SymbolicBlockMatrix sym_structure;
  /// @returns bool whether the decomposition was successful.
  bool ldlt_in_place_impl(SignMatrix &sign) {
    if (!sym_structure.performed_llt) {
      assert(false && "Block structure was not analyzed yet.");
      return false;
    }
    const isize nblocks = sym_structure.nsegments();
    const isize n = mat.rows();
    if ((nblocks == 0) || (n <= 1)) {
      return true;
    }

    const isize bs = sym_structure.segment_lens[0];
    const isize rem = n - bs;
    MatrixRef l00 = mat.block(0, 0, bs, bs);
    MatrixRef l11 = mat.block(bs, bs, rem, rem);
    const auto d0 = l00.diagonal();
    const auto d0_inv = d0.asDiagonal().inverse();

    MatrixRef work_tr = mat.block(0, bs, bs, rem);
    Eigen::Transpose<MatrixRef> work = work_tr.transpose();

    switch (sym_structure(0, 0)) {
    case Zero:
    case TriU:
    case Dense:
      return false;

    case TriL: {
      // compute l00
      backend::dense_ldlt_in_place(l00, sign);

      isize offset = bs;

      for (isize i = 1; i < nblocks; ++i) {
        const isize bsi = sym_structure.segment_lens[i];
        MatrixRef li0 = mat.block(offset, 0, bsi, bs);
        Eigen::Block<decltype(work)> li0_copy =
            work.block(offset - bs, 0, bsi, bs);

        switch (sym_structure(i, 0)) {
        case TriL:
          return false;
        case TriU: {
          auto li0_u = li0.template triangularView<Eigen::Upper>();
          // PERF: replace li0.T by li0_tl
          // auto li0_tl = li0.transpose().template
          // triangularView<Eigen::Lower>();
          l00.template triangularView<Eigen::UnitLower>().solveInPlace(
              li0.transpose());
          li0_copy.template triangularView<Eigen::Upper>() = li0_u;
          li0_u = li0 * d0_inv;
          break;
        }
        /// TODO: make this smarter for Diag
        case Diag:
        case Dense: {
          l00.template triangularView<Eigen::UnitLower>().solveInPlace(
              li0.transpose());
          li0_copy = li0;
          li0 = li0 * d0_inv;
          break;
        }
        case Zero:
          break;
        }
        offset += bsi;
      }

      break;
    }
    case Diag: {
      // l00 is unchanged
      for (isize k = 0; k < bs; k++) {
        update_sign_matrix(sign, l00(k, k));
      }
      isize offset = bs;

      for (isize i = 1; i < nblocks; ++i) {
        const isize bsi = sym_structure.segment_lens[i];
        MatrixRef li0 = mat.block(offset, 0, bsi, bs);
        Eigen::Block<decltype(work)> li0_copy =
            work.block(offset - bs, 0, bsi, bs);

        switch (sym_structure(i, 0)) {
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
          li0_u = li0 * d0_inv;
          break;
        }
        case Dense: {
          li0_copy = li0;
          li0 = li0 * d0_inv;
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
      const isize bsi = sym_structure.segment_lens[i];
      Eigen::Block<MatrixRef> li0 = mat.block(offset_i, 0, bsi, bs);
      Eigen::Block<decltype(work)> li0_prev =
          work.block(offset_i - bs, 0, bsi, bs);

      /// WARNING: target_ii CONTAINS LAST ROW OF "WORK"
      Eigen::Block<MatrixRef> target_ii =
          mat.block(offset_i, offset_i, bsi, bsi);

      // target_ii -= li0 * li0_prev.T;
      /// TODO: FIX ALIASING HERE, target_ii CONTAINS COEFFS FROM li0_prev
      backend::gemmt(target_ii, li0, li0_prev, sym_structure(i, 0),
                     sym_structure(i, 0), Scalar(-1));

      isize offset_j = offset_i + bsi;
      for (isize j = i + 1; j < nblocks; ++j) {
        // target_ji -= lj0 * li0_prev.T;

        const isize bsj = sym_structure.segment_lens[j];
        Eigen::Block<MatrixRef> lj0 = mat.block(offset_j, 0, bsj, bs);
        Eigen::Block<MatrixRef> target_ji =
            mat.block(offset_j, offset_i, bsj, bsi);

        backend::gemmt(target_ji, lj0, li0_prev, sym_structure(j, 0),
                       sym_structure(i, 0), Scalar(-1));

        offset_j += bsj;
      }

      offset_i += bsi;
    }

    return block_impl{
        l11,
        sym_structure.submatrix(1, nblocks - 1),
    }
        .ldlt_in_place_impl(sign);
  }
};

} // namespace backend

/// @brief Block sparsity-aware LDLT factorization algorithm.
/// @details  This struct owns the data of the SymbolicBlockMatrix given as
/// input.
/// The member function findSparsifyingPermutation() uses a heuristic
/// (for now a brute-force search) to find a sparsity-maximizing permutation of
/// the blocks in the input matrix.
/// updateBlockPermutationMatrix() updates the permutation matrix according to
/// the stored block-wise permutation indices.
///
/// @warning  The underlying block-wise structure is assumed to be invariant
/// over the lifetime of this object when calling compute(). A change
/// of structure should lead to recalculating the expected sparsity pattern of
/// the factorization, and even recomputing the sparsity-optimal permutation.
template <typename _Scalar> struct BlockLDLT : ldlt_base<_Scalar> {
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ldlt_base<Scalar>;
  using DView = typename Base::DView;
  using Traits = LDLT_Traits<MatrixXs, Eigen::Lower>;
  using PermutationType =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, isize>;
  using PermIdxType = Eigen::Matrix<isize, Eigen::Dynamic, 1>;
  using BlockTriL = TriangularBlockMatrix<const MatrixXs, Eigen::UnitLower>;
  using BlockTriU =
      TriangularBlockMatrix<const typename MatrixXs::AdjointReturnType,
                            Eigen::UnitUpper>;

protected:
  MatrixXs m_matrix;
  SymbolicBlockMatrix m_structure;
  SymbolicBlockMatrix m_struct_tr;
  PermutationType m_permutation;
  using Base::m_info;
  using Base::m_sign;
  std::vector<isize> m_perm;
  std::vector<isize> m_perm_inv;
  std::vector<isize> m_iwork;
  std::vector<isize> m_start_idx;

  BlockLDLT &updateBlockPermutationMatrix(SymbolicBlockMatrix const &in);

public:
  /// @brief  The constructor copies the input matrix @param mat and symbolic
  /// block pattern @param structure.
  BlockLDLT(isize size, SymbolicBlockMatrix const &structure)
      : Base(), m_matrix(size, size), m_structure(structure.copy()),
        m_permutation(size), m_perm(nblocks()), m_perm_inv(nblocks()),
        m_iwork(nblocks()), m_start_idx(nblocks()) {
    std::iota(m_perm.begin(), m_perm.end(), isize(0));
    m_permutation.setIdentity();
    m_struct_tr = m_structure.transpose();
  }

  BlockLDLT(BlockLDLT const &other)
      : BlockLDLT(other.m_matrix.rows(), other.m_structure) {
    m_matrix = other.m_matrix;
    m_permutation = other.m_permutation;
    m_info = other.m_info;
    m_perm = other.m_perm;
    m_perm_inv = other.m_perm_inv;
    m_iwork = other.m_iwork;
    m_start_idx = other.m_start_idx;
  }

  /// Compute indices indicating where blocks start
  void computeStartIndices(const SymbolicBlockMatrix &in) {
    m_start_idx[0] = 0;
    for (usize i = 0; i < nblocks() - 1; ++i) {
      m_start_idx[i + 1] = m_start_idx[i] + in.segment_lens[i];
    }
  }

  ~BlockLDLT() {
    delete[] m_structure.m_data;
    delete[] m_structure.segment_lens;
    delete[] m_struct_tr.m_data;
    delete[] m_struct_tr.segment_lens;
    m_structure.m_data = nullptr;
    m_structure.segment_lens = nullptr;
  }

  /// @returns a reference to the symbolic representation of the block-matrix.
  inline const SymbolicBlockMatrix &structure() const { return m_structure; }
  inline void print_sparsity() const { print_sparsity_pattern(m_structure); }

  /// @brief Analyze and factorize the block structure, if not done already.
  inline bool analyzePattern();

  usize nblocks() const { return usize(m_structure.segments_count); }

  /// Calls updateBlockPermutationMatrix
  void setBlockPermutation(isize const *new_perm = nullptr);

  auto blockPermIndices() -> std::vector<isize> & { return m_perm; }

  /// @brief Find a sparsity-maximizing permutation of the blocks. This will
  /// also compute the symbolic factorization.
  BlockLDLT &findSparsifyingPermutation();

  inline const PermutationType &permutationP() const { return m_permutation; }

  MatrixXs reconstructedMatrix() const override;

  inline typename Traits::MatrixL matrixL() const {
    return Traits::getL(m_matrix);
  }

  inline typename Traits::MatrixU matrixU() const {
    return Traits::getU(m_matrix);
  }

  inline DView vectorD() const override {
    return Base::diag_view_impl(m_matrix);
  }

  /// Solve for the right-hand side in-place.
  template <typename Derived>
  bool solveInPlace(Eigen::MatrixBase<Derived> &b) const;

  const MatrixXs &matrixLDLT() const override { return m_matrix; }

  inline void compute() {
    m_info =
        backend::block_impl<Scalar>{m_matrix, m_structure}.ldlt_in_place_impl(
            m_sign)
            ? Eigen::Success
            : Eigen::NumericalIssue;
  }

  /// Sets the input matrix to @p mat, performs the permutation and runs the
  /// algorithm.
  BlockLDLT &compute(const ConstMatrixRef &mat) override {
    assert(mat.rows() == mat.cols());
    m_matrix.conservativeResizeLike(mat);
    auto mat_coeff = [&](isize i, isize j) {
      return i >= j ? mat(i, j) : mat(j, i);
    };
    auto n = mat.rows();
    auto indices = permutationP().indices();
    // by column
    for (isize j = 0; j < n; ++j) {
      auto pj = indices[j];
      // by line starting at j
      for (isize i = j; i < n; ++i) {
        m_matrix(i, j) = mat_coeff(indices[i], pj);
      }
    }
    // m_matrix.noalias() = permutationP() * m_matrix;
    // m_matrix.noalias() = m_matrix * permutationP().transpose();
    // do not re-run analysis
    compute();

    return *this;
  }
};

} // namespace linalg
} // namespace nlp
} // namespace proxsuite

#include "./block-ldlt.hxx"

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "./block-ldlt.txx"
#endif
