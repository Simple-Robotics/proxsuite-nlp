/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @brief  Definition for matrix "kind" enums.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

namespace proxnlp {
namespace linalg {

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

} // namespace linalg
} // namespace proxnlp
