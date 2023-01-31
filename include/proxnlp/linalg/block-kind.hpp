/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @brief  Definition for matrix "kind" enums.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include <Eigen/Core>

namespace proxnlp {
namespace linalg {

namespace {
using isize = Eigen::Index;
}

/// Kind of matrix block: zeros, diagonal, lower/upper triangular or dense.
enum BlockKind {
  /// All entries in the block are zero.
  Zero,
  /// The block is diagonal.
  Diag,
  /// The block is lower-triangular.
  TriL,
  /// The block is upper-triangular.
  TriU,
  /// There is no known prior structure; assume a dense block.
  Dense,
};

/// BlockKind of the transpose of a matrix.
BlockKind trans(BlockKind a) noexcept;

/// BlockKind of the addition of two matrices - given by their BlockKind.
BlockKind add(BlockKind a, BlockKind b) noexcept;

/// BlockKind of the product of two matrices.
BlockKind mul(BlockKind a, BlockKind b) noexcept;

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
