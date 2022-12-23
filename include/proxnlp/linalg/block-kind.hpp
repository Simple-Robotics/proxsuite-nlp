#pragma once

namespace proxnlp {
namespace block_chol {

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

} // namespace block_chol
} // namespace proxnlp
