/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
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

namespace proxnlp {
/// @brief	Block-wise Cholesky and LDLT factorisation routines.
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
  /// Flag stating whether the block structure was successfully analyzed.
  /// This should be checked when attempting to factorize.
  bool performed_llt = false;

  SymbolicBlockMatrix() = delete;
  /// Shallow copy constructor.
  SymbolicBlockMatrix(SymbolicBlockMatrix const &other) = default;
  SymbolicBlockMatrix &operator=(SymbolicBlockMatrix const &other) = delete;

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
};

/// TODO: print triangles for triangular blocks
void print_sparsity_pattern(const SymbolicBlockMatrix &smat) noexcept;

/// Deep copy of a SymbolicBlockMatrix, possibily with a permutation.
void symbolic_deep_copy(const SymbolicBlockMatrix &in, SymbolicBlockMatrix &out,
                        isize const *perm = nullptr) noexcept;

namespace backend {

/// At the end of the execution, @param a contains
/// the lower-triangular matrix \f$L\f$ in the LDLT decomposition.
/// More precisely: a stores L -sans its diagonal which is all ones.
/// The diagonal of @param a contains the diagonal matrix @f$D@f$.
inline bool ldlt_in_place_unblocked(MatrixRef a) {
  const isize n = a.rows();
  if (n <= 1) {
    return true;
  }

  isize j = 0;
  while (true) {
    auto l10 = a.row(j).head(j);
    auto d0 = a.diagonal().head(j);
    auto work = a.col(n - 1).head(j);

    work = l10.transpose().cwiseProduct(d0);
    a(j, j) -= work.dot(l10);

    if (j + 1 == n) {
      return true;
    }

    const isize rem = n - j - 1;

    auto l20 = a.bottomLeftCorner(rem, j);
    auto l21 = a.col(j).tail(rem);

    l21.noalias() -= l20 * work;
    l21 *= 1 / a(j, j);
    ++j;
  }
}

/// A recursive, in-place implementation of the LDLT decomposition.
/// To be applied to dense blocks.
inline bool dense_ldlt_in_place(MatrixRef const &a) {
  isize n = a.rows();
  if (n <= 128) {
    return backend::ldlt_in_place_unblocked(a);
  } else {
    isize bs = (n + 1) / 2;
    isize rem = n - bs;

    auto a_mut = a.const_cast_derived();
    MatrixRef l00 = a_mut.block(0, 0, bs, bs);
    MatrixRef l10 = a_mut.block(bs, 0, rem, bs);
    MatrixRef l11 = a_mut.block(bs, bs, rem, rem);

    backend::dense_ldlt_in_place(l00);
    auto d0 = l00.diagonal();

    l00.transpose()
        .template triangularView<Eigen::UnitUpper>()
        .template solveInPlace<Eigen::OnTheRight>(l10);

    MatrixRef work = a_mut.block(0, rem, rem, bs);
    work = l10;
    l10 = l10 * d0.asDiagonal().inverse();

    l11.template triangularView<Eigen::Lower>() -= l10 * work.transpose();

    return backend::dense_ldlt_in_place(l11);
  }
}

using Eigen::internal::LDLT_Traits;

/// Taking the decomposed LDLT matrix @param mat, solve the original linear
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

inline void dense_ldlt_reconstruct(ConstMatrixRef const &mat, MatrixRef res) {
  typedef LDLT_Traits<ConstMatrixRef, Eigen::Lower> Traits;
  res = Traits::getU(mat) * res;

  auto vecD = mat.diagonal();
  res = vecD.asDiagonal() * res;

  res = Traits::getL(mat) * res;
}

#if true

#include "linalg/gemmt-v1.hpp"

#else

#include "linalg/gemmt-v2.hpp"

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

/// @brief  A fast, recursive divide-and-conquer LDLT algorithm.
struct DenseLDLT {
  using MatrixXs = MatrixRef::PlainMatrix;

  DenseLDLT() = default;
  explicit DenseLDLT(isize n) : m_matrix(n, n) { m_matrix.setZero(); }
  explicit DenseLDLT(MatrixRef a) : m_matrix(a) {
    m_info = backend::dense_ldlt_in_place(m_matrix) ? Eigen::Success
                                                    : Eigen::NumericalIssue;
  }

  DenseLDLT &compute(MatrixRef mat) {
    m_matrix = mat;
    m_info = backend::dense_ldlt_in_place(m_matrix) ? Eigen::Success
                                                    : Eigen::NumericalIssue;
    return *this;
  }

  const MatrixXs &matrixLDLT() const { return m_matrix; }

  Eigen::ComputationInfo info() const { return m_info; }

  void solveInPlace(MatrixRef b) const {
    backend::dense_ldlt_solve_in_place(m_matrix, b);
  }

  MatrixXs reconstructedMatrix() const {
    MatrixXs res(m_matrix.rows(), m_matrix.cols());
    res.setIdentity();
    backend::dense_ldlt_reconstruct(m_matrix, res);
    return res;
  }

  Eigen::Diagonal<const MatrixXs> vectorD() const {
    return m_matrix.diagonal();
  }

protected:
  MatrixXs m_matrix;
  Eigen::ComputationInfo m_info;
  bool permutate = false;
};

namespace backend {

/// Implementation struct for the recursive block LDLT algorithm.
struct block_impl {
  using MatrixXs = MatrixRef::PlainMatrix;
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
    auto d0 = l00.diagonal();

    // temp workspace -> upper triangle of matrix, filled with "garbage"
    /// TODO: FIX ALLOCATION HERE, USING work CREATES ALIASING LATER IN CALL TO
    /// GEMMT
    MatrixXs work = mat.block(0, rem, rem, bs);

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
        MatrixRef li0_copy = work.block(offset - bs, 0, bsi, bs);

        switch (sym_structure(i, 0)) {
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
        MatrixRef li0_copy = work.block(offset - bs, 0, bsi, bs);

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
      const MatrixRef li0 = mat.block(offset_i, 0, bsi, bs);
      const MatrixRef li0_prev = work.block(offset_i - bs, 0, bsi, bs);

      /// WARNING: target_ii CONTAINS LAST ROW OF "WORK"
      MatrixRef target_ii = mat.block(offset_i, offset_i, bsi, bsi);

      // target_ii -= li0 * li0_prev.T;
      /// TODO: FIX ALIASING HERE, target_ii CONTAINS COEFFS FROM li0_prev
      backend::gemmt(target_ii, li0, li0_prev, sym_structure(i, 0),
                     sym_structure(i, 0), Scalar(-1));

      isize offset_j = offset_i + bsi;
      for (isize j = i + 1; j < nblocks; ++j) {
        // target_ji -= lj0 * li0_prev.T;

        isize bsj = sym_structure.segment_lens[j];
        const MatrixRef lj0 = mat.block(offset_j, 0, bsj, bs);
        MatrixRef target_ji = mat.block(offset_j, offset_i, bsj, bsi);

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
/// updateBlockPermutMatrix() updates the permutation matrix according to the
/// stored block-wise permutation indices. Calling permute() will perform the
/// permutation on the input matrix.
///
/// @warning  The underlying block-wise structure is assumed to be invariant
/// over the lifetime of this object when calling compute(). A change
/// of structure should lead to recalculating the expected sparsity pattern of
/// the factorization, and even recomputing the sparsity-optimal permutation.
struct BlockLDLT {
  using MatrixXs = MatrixRef::PlainMatrix;
  using Traits = backend::LDLT_Traits<MatrixXs, Eigen::Lower>;
  using PermutationType =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, isize>;

protected:
  MatrixXs m_matrix;
  SymbolicBlockMatrix m_structure;
  PermutationType m_permutation;
  Eigen::ComputationInfo m_info;
  isize *perm;
  isize *iwork;
  isize *idx_cumul;

  std::size_t nblocks() const {
    return std::size_t(m_structure.segments_count);
  }

public:
  /// @brief  The constructor copies the input matrix @param mat and symbolic
  /// block pattern @param structure.
  explicit BlockLDLT(isize n, SymbolicBlockMatrix const &structure)
      : m_matrix(n, n), m_structure(structure.copy()), m_permutation(n),
        perm(new isize[nblocks()]), iwork(new isize[nblocks()]),
        idx_cumul(new isize[nblocks()]) {
    std::iota(perm, perm + nblocks(), isize(0));
  }

  /// @copydoc BlockLDLT()
  /// @todo run the algorithm when in this constructor. Perhaps provide flags to
  /// decide whether to compute/use permutations.
  BlockLDLT(MatrixRef mat, SymbolicBlockMatrix const &structure)
      : m_matrix(mat), m_structure(structure.copy()), m_permutation(mat.rows()),
        perm(new isize[nblocks()]), iwork(new isize[nblocks()]),
        idx_cumul(new isize[nblocks()]) {
    std::iota(perm, perm + nblocks(), isize(0));
  }

  ~BlockLDLT() {
    delete[] perm;
    delete[] iwork;
    delete[] idx_cumul;
    delete[] m_structure.data;
    delete[] m_structure.segment_lens;
    m_structure.data = nullptr;
    m_structure.segment_lens = nullptr;
  }

  Eigen::ComputationInfo info() const { return m_info; }

  /// @returns a reference to the symbolic representation of the block-matrix.
  inline const SymbolicBlockMatrix &structure() const { return m_structure; }

  /// @brief Analyze and factorize the block structure, if not done already.
  inline bool performAnalysis() {
    if (m_structure.performed_llt)
      return true;
    return m_structure.llt_in_place();
  }

  void setPermutation(isize const *new_perm = nullptr) {
    auto in = m_structure.copy();
    const isize n = m_structure.nsegments();
    if (new_perm != nullptr)
      std::copy_n(new_perm, n, perm);
    m_structure.performed_llt = false;
    symbolic_deep_copy(in, m_structure, perm);
    performAnalysis();
  }

  isize *blockPermIndices() { return perm; }

  /// @brief Find a sparsity-maximizing permutation of the blocks. This will
  /// also compute the symbolic factorization.
  void findSparsifyingPermutation() {
    auto in = m_structure.copy();
    m_structure.brute_force_best_permutation(in, perm, iwork);
    symbolic_deep_copy(in, m_structure, perm);
    performAnalysis();
  }

  inline const PermutationType &permutationP() const { return m_permutation; }

  void updateBlockPermutMatrix(SymbolicBlockMatrix const &in) {
    const isize *row_segs = in.segment_lens;
    const isize nblocks = in.nsegments();
    using IndicesType = PermutationType::IndicesType;
    IndicesType &indices = m_permutation.indices();
    isize idx = 0;
    for (isize i = 0; i < nblocks; ++i) {
      idx_cumul[i] = idx;
      idx += row_segs[i];
    }

    idx = 0;
    for (isize i = 0; i < nblocks; ++i) {
      auto len = row_segs[perm[i]];
      auto s = indices.segment(idx, len);
      isize i0 = idx_cumul[perm[i]];
      s.setLinSpaced(i0, i0 + len - 1);
      idx += len;
    }
    m_permutation = m_permutation.transpose();
  }

  MatrixXs reconstructedMatrix() const {
    MatrixXs res(m_matrix.rows(), m_matrix.cols());
    res = permutationP();
    backend::dense_ldlt_reconstruct(m_matrix, res);
    res = permutationP().transpose() * res;
    return res;
  }

  Traits::MatrixL matrixL() const { return Traits::getL(m_matrix); }

  Traits::MatrixU matrixU() const { return Traits::getU(m_matrix); }

  /// TODO: make block-sparse variant of solveInPlace()
  bool solveInPlace(MatrixRef b) const {
    b = permutationP() * b;
    bool flag = backend::dense_ldlt_solve_in_place(m_matrix, b);
    b = permutationP().transpose() * b;
    return flag;
  }

  const MatrixXs &matrixLDLT() const { return m_matrix; }

  void permute() {
    m_matrix = permutationP() * m_matrix;
    m_matrix = m_matrix * permutationP().transpose();
  }

  void compute() {
    m_info = backend::block_impl{m_matrix, m_structure}.ldlt_in_place_impl()
                 ? Eigen::Success
                 : Eigen::NumericalIssue;
  }

  BlockLDLT &compute(MatrixRef mat) {
    m_matrix = mat;
    // do not re-run analysis
    compute();

    return *this;
  }
};

} // namespace block_chol
} // namespace proxnlp
