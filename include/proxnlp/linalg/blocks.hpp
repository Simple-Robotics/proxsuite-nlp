/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @brief Routines for block-sparse (notably, KKT-type) matrix LDLT
/// factorisation.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/linalg/dense.hpp"
#include "proxnlp/linalg/block-kind.hpp"

#include <algorithm>
#include <numeric>

namespace proxnlp {
namespace linalg {

// fwd declaration
struct SymbolicBlockMatrix;

// fwd declaration
template <typename Scalar> struct BlockLDLT;

// fwd declaration
template <typename MatType, int Mode> struct TriangularBlockMatrix;

/// @brief    Symbolic representation of the sparsity structure of a (square)
/// block matrix.
/// @details  This struct describes the block-wise layout of a matrix, in
/// row-major format.
struct SymbolicBlockMatrix {
  BlockKind *data;
  isize *segment_lens;
  isize segments_count;
  isize outer_stride;
  /// Flag stating whether the block structure was successfully analyzed.
  /// This should be checked when attempting to factorize.
  bool performed_llt = false;

  SymbolicBlockMatrix() = delete;
  /// Shallow copy constructor.
  SymbolicBlockMatrix(SymbolicBlockMatrix const &other) = default;
  SymbolicBlockMatrix &operator=(SymbolicBlockMatrix const &other) = default;

  /// Deep copy.
  SymbolicBlockMatrix copy() const;

  isize nsegments() const noexcept { return segments_count; }
  isize size() const noexcept { return segments_count * outer_stride; }
  BlockKind *ptr(isize i, isize j) const noexcept {
    return data + (i + j * outer_stride);
  }

  /// Get the symbolic submatrix of size (n, n) starting from the block in
  /// position (i, i).
  SymbolicBlockMatrix submatrix(isize i, isize n) const noexcept;
  /// Get a reference to the block in position (i, j).
  BlockKind &operator()(isize i, isize j) const noexcept { return *ptr(i, j); }

  /// Brute-force search of the best permutation possible in the block matrix,
  /// with respect to the final sparsity of the LLT decomposition.
  /// The struct instance *this will be the result.
  /// @param in    the input matrix to analyze.
  /// @param iwork workspace; has length `in.nsegments()`.
  Eigen::ComputationInfo
  brute_force_best_permutation(SymbolicBlockMatrix const &in, isize *best_perm,
                               isize *iwork);
  bool check_if_symmetric() const noexcept;
  isize count_nnz() const noexcept;
  /// Perform symbolic block-wise LLT decomposition;
  /// the output sparsity pattern should be that of the matrix \f$L\f$
  /// of the Cholesky decomposition.
  bool llt_in_place() noexcept;

  SymbolicBlockMatrix transpose() const {
    const auto &self = *this;
    SymbolicBlockMatrix s2(copy());
    for (isize i = 0; i < nsegments(); ++i) {
      for (isize j = 0; j < nsegments(); ++j) {
        s2(i, j) = trans(self(j, i));
      }
    }
    return s2;
  }
};

/// TODO: print triangles for triangular blocks
void print_sparsity_pattern(const SymbolicBlockMatrix &smat) noexcept;

/// Deep copy of a SymbolicBlockMatrix, possibily with a permutation.
void symbolic_deep_copy(const SymbolicBlockMatrix &in, SymbolicBlockMatrix &out,
                        isize const *perm = nullptr) noexcept;

} // namespace linalg
} // namespace proxnlp

#include "proxnlp/linalg/gemmt.hpp"

namespace proxnlp {
namespace linalg {
namespace backend {

template <typename Scalar> struct gemmt {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  template <typename DstDerived, typename LhsDerived, typename RhsDerived>
  inline static void run(Eigen::MatrixBase<DstDerived> &dst,
                         Eigen::MatrixBase<LhsDerived> const &lhs,
                         Eigen::MatrixBase<RhsDerived> const &rhs,
                         BlockKind lhs_kind, BlockKind rhs_kind, Scalar alpha) {
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
};

/// Implementation struct for the recursive block LDLT algorithm.
template <typename Scalar> struct block_impl {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  MatrixRef mat;
  SymbolicBlockMatrix sym_structure;
  /// @returns bool whether the decomposition was successful.
  bool ldlt_in_place_impl() {
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

    MatrixRef work_tr = mat.block(0, bs, bs, rem);
    Eigen::Transpose<MatrixRef> work = work_tr.transpose();

    switch (sym_structure(0, 0)) {
    case Zero:
    case TriU:
    case Dense:
      return false;

    case TriL: {
      // compute l00
      backend::dense_ldlt_in_place(l00);

      isize offset = bs;

      for (isize i = 1; i < nblocks; ++i) {
        const isize bsi = sym_structure.segment_lens[i];
        MatrixRef li0 = mat.block(offset, 0, bsi, bs);
        Eigen::Block<decltype(work)> li0_copy =
            work.block(offset - bs, 0, bsi, bs);

        switch (sym_structure(i, 0)) {
        case Diag:
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
        offset += bsi;
      }

      break;
    }
    case Diag: {
      // l00 is unchanged
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
      const isize bsi = sym_structure.segment_lens[i];
      Eigen::Block<MatrixRef> li0 = mat.block(offset_i, 0, bsi, bs);
      Eigen::Block<decltype(work)> li0_prev =
          work.block(offset_i - bs, 0, bsi, bs);

      /// WARNING: target_ii CONTAINS LAST ROW OF "WORK"
      Eigen::Block<MatrixRef> target_ii =
          mat.block(offset_i, offset_i, bsi, bsi);

      // target_ii -= li0 * li0_prev.T;
      /// TODO: FIX ALIASING HERE, target_ii CONTAINS COEFFS FROM li0_prev
      backend::gemmt<Scalar>::run(target_ii, li0, li0_prev, sym_structure(i, 0),
                                  sym_structure(i, 0), Scalar(-1));

      isize offset_j = offset_i + bsi;
      for (isize j = i + 1; j < nblocks; ++j) {
        // target_ji -= lj0 * li0_prev.T;

        const isize bsj = sym_structure.segment_lens[j];
        Eigen::Block<MatrixRef> lj0 = mat.block(offset_j, 0, bsj, bs);
        Eigen::Block<MatrixRef> target_ji =
            mat.block(offset_j, offset_i, bsj, bsi);

        backend::gemmt<Scalar>::run(target_ji, lj0, li0_prev,
                                    sym_structure(j, 0), sym_structure(i, 0),
                                    Scalar(-1));

        offset_j += bsj;
      }

      offset_i += bsi;
    }

    return block_impl{
        l11,
        sym_structure.submatrix(1, nblocks - 1),
    }
        .ldlt_in_place_impl();
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
/// the stored block-wise permutation indices. Calling permutate() will perform
/// the permutation on the input matrix.
///
/// @warning  The underlying block-wise structure is assumed to be invariant
/// over the lifetime of this object when calling compute(). A change
/// of structure should lead to recalculating the expected sparsity pattern of
/// the factorization, and even recomputing the sparsity-optimal permutation.
template <typename Scalar> struct BlockLDLT : ldlt_base<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ldlt_base<Scalar>;
  using Traits = LDLT_Traits<MatrixXs, Eigen::Lower>;
  using PermutationType =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, isize>;
  using BlockTriL = TriangularBlockMatrix<const MatrixXs, Eigen::UnitLower>;
  using BlockTriU =
      TriangularBlockMatrix<const typename MatrixXs::AdjointReturnType,
                            Eigen::UnitUpper>;

protected:
  MatrixXs m_matrix;
  SymbolicBlockMatrix m_structure;
  SymbolicBlockMatrix m_struct_transposed;
  PermutationType m_permutation;
  using Base::m_info;
  isize *m_perm;
  isize *m_iwork;
  isize *m_idx;

public:
  /// @brief  The constructor copies the input matrix @param mat and symbolic
  /// block pattern @param structure.
  BlockLDLT(isize size, SymbolicBlockMatrix const &structure)
      : Base(), m_matrix(size, size), m_structure(structure.copy()),
        m_struct_transposed(m_structure.transpose()), m_permutation(size),
        m_perm(new isize[nblocks()]), m_iwork(new isize[nblocks()]),
        m_idx(new isize[nblocks()]) {
    std::iota(m_perm, m_perm + nblocks(), isize(0));
    m_permutation.setIdentity();
  }

  /// @copydoc BlockLDLT()
  /// @todo run the algorithm when in this constructor. Perhaps provide flags to
  /// decide whether to compute/use permutations.
  BlockLDLT(MatrixRef mat, SymbolicBlockMatrix const &structure)
      : m_matrix(mat), m_structure(structure.copy()), m_permutation(mat.rows()),
        m_perm(new isize[nblocks()]), m_iwork(new isize[nblocks()]),
        m_idx(new isize[nblocks()]) {
    findSparsifyingPermutation();
    updateBlockPermutationMatrix(structure);
    compute(mat);
  }

  BlockLDLT(BlockLDLT const &other)
      : BlockLDLT(other.m_matrix.rows(), other.m_structure) {
    m_matrix = other.m_matrix;
    m_permutation = other.m_permutation;
    m_info = other.m_info;
    std::copy_n(other.m_perm, nblocks(), m_perm);
    std::copy_n(other.m_iwork, nblocks(), m_iwork);
  }

  ~BlockLDLT() {
    delete[] m_perm;
    delete[] m_iwork;
    delete[] m_idx;
    delete[] m_structure.data;
    delete[] m_structure.segment_lens;
    delete[] m_struct_transposed.data;
    delete[] m_struct_transposed.segment_lens;
    m_structure.data = nullptr;
    m_structure.segment_lens = nullptr;
    m_struct_transposed.data = nullptr;
    m_struct_transposed.segment_lens = nullptr;
  }

  /// @returns a reference to the symbolic representation of the block-matrix.
  inline const SymbolicBlockMatrix &structure() const { return m_structure; }

  /// @brief Analyze and factorize the block structure, if not done already.
  inline bool analyzePattern() {
    if (m_structure.performed_llt)
      return true;
    return m_structure.llt_in_place();
  }

  std::size_t nblocks() const {
    return std::size_t(m_structure.segments_count);
  }

  void setPermutation(isize const *new_perm = nullptr);

  isize *blockPermIndices() { return m_perm; }

  /// @brief Find a sparsity-maximizing permutation of the blocks. This will
  /// also compute the symbolic factorization.
  BlockLDLT &findSparsifyingPermutation() {
    auto in = m_structure.copy();
    m_structure.brute_force_best_permutation(in, m_perm, m_iwork);
    symbolic_deep_copy(in, m_structure, m_perm);
    analyzePattern();
    return *this;
  }

  inline const PermutationType &permutationP() const { return m_permutation; }

  BlockLDLT &updateBlockPermutationMatrix(SymbolicBlockMatrix const &in);

  MatrixXs reconstructedMatrix() const override;

  inline typename Traits::MatrixL matrixL() const {
    return Traits::getL(m_matrix);
  }

  inline typename Traits::MatrixU matrixU() const {
    return Traits::getU(m_matrix);
  }

  inline Eigen::Diagonal<const MatrixXs> vectorD() const override {
    return m_matrix.diagonal();
  }

  /// TODO: make block-sparse variant of solveInPlace()
  bool solveInPlace(MatrixRef b) const override;

  const MatrixXs &matrixLDLT() const override { return m_matrix; }

  inline void permutate() {
    m_matrix.noalias() = permutationP() * m_matrix;
    m_matrix.noalias() = m_matrix * permutationP().transpose();
  }

  void compute() {
    m_info =
        backend::block_impl<Scalar>{m_matrix, m_structure}.ldlt_in_place_impl()
            ? Eigen::Success
            : Eigen::NumericalIssue;
  }

  /// Sets the input matrix to @p mat, performs the permutation and runs the
  /// algorithm.
  BlockLDLT &compute(const MatrixRef &mat) override {
    m_matrix = mat;
    // do not re-run analysis
    permutate();
    compute();

    return *this;
  }
};

} // namespace linalg
} // namespace proxnlp

#include "proxnlp/linalg/block-triangular.hpp"
#include "proxnlp/linalg/blocks.hxx"
