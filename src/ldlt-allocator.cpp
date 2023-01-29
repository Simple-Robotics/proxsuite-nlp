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

  BlockKind *blocks = new BlockKind[(uint)(nblocks * nblocks)];
  isize *segment_lens = new isize[(uint)nblocks];

  for (uint i = 0; i < nprim_blocks; ++i) {
    segment_lens[i] = dims_primal[i];
  }
  for (uint i = 0; i < ndual_blocks; ++i) {
    segment_lens[i + nprim_blocks] = dims_dual[i];
  }

  SymbolicBlockMatrix structure{blocks, segment_lens, nblocks, nblocks, false};

  // default structure: primal blocks are dense, others are sparse

  for (isize i = 0; i < nprim_blocks; ++i) {
    for (isize j = 0; j < nprim_blocks; ++j) {
      structure(i, j) = BlockKind::Dense;
    }
  }

  for (isize i = nprim_blocks; i < nblocks; ++i) {
    // first col/row
    structure(i, nprim_blocks) = BlockKind::Dense;
    structure(nprim_blocks, i) = BlockKind::Dense;

    for (isize j = nprim_blocks; j < nblocks; ++j) {
      structure(i, j) = BlockKind::Zero;
    }

    // diag
    structure(i, i) = BlockKind::Diag;
  }
  return structure;
}

} // namespace proxnlp
