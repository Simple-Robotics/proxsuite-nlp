/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxnlp/ldlt-allocator.hpp"

namespace proxnlp {

SymbolicBlockMatrix
create_default_block_structure(const std::vector<isize> &dims_primal,
                               const std::vector<isize> &dims_dual) {
  using linalg::BlockKind;

  isize nprim_blocks = (isize)dims_primal.size();
  isize ndual_blocks = (isize)dims_dual.size();
  isize nblocks = nprim_blocks + ndual_blocks;

  SymbolicBlockMatrix structure(nblocks, nblocks);
  isize *segment_lens = structure.segment_lens;

  for (uint i = 0; i < nprim_blocks; ++i) {
    segment_lens[i] = dims_primal[i];
  }
  for (uint i = 0; i < ndual_blocks; ++i) {
    segment_lens[i + nprim_blocks] = dims_dual[i];
  }

  // default structure: primal blocks are dense, others are sparse

  for (isize i = 0; i < nprim_blocks; ++i) {
    for (isize j = 0; j < nprim_blocks; ++j) {
      structure(i, j) = linalg::Dense;
    }
  }

  // jacobian blocks: assumed dense
  for (isize i = 0; i < nprim_blocks; ++i) {
    for (isize j = nprim_blocks; j < nblocks; ++j) {
      structure(i, j) = linalg::Dense;
      structure(j, i) = linalg::Dense;
    }
  }

  for (isize i = nprim_blocks; i < nblocks; ++i) {
    // diagonal blocks are diagonal
    structure(i, i) = linalg::Diag;

    // off-diagonal blocks are zero
    for (isize j = nprim_blocks; j < nblocks; ++j) {
      if (i != j)
        structure(i, j) = linalg::Zero;
    }
  }
  return structure;
}

} // namespace proxnlp
