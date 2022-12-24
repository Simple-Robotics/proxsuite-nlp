/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "linalg/blocks.hpp"

#include <boost/variant.hpp>

namespace proxnlp {

using boost::variant;

/// @brief A variant type for all available
template <typename Scalar>
using LDLT_variant =
    variant<block_chol::DenseLDLT<Scalar>, block_chol::BlockLDLT<Scalar>>;

template <typename Scalar>
block_chol::BlockLDLT<Scalar>
initialize_block_ldlt_from_structure(long ndx, std::vector<long> nduals) {
  using block_chol::BlockKind;
  using block_chol::isize;
  using block_chol::SymbolicBlockMatrix;
  const isize nblocks = 1 + (isize)nduals.size();

  isize tot_size = ndx;
  for (const long &s : nduals) {
    tot_size += s;
  }

  std::vector<BlockKind> blocks((std::size_t)(nblocks * nblocks));
  std::vector<long> seg_lens = nduals;
  seg_lens.insert(seg_lens.begin(), ndx);

  SymbolicBlockMatrix structure{blocks.data(), seg_lens.data(), nblocks,
                                nblocks, false};
  // default structure: first column/row is all dense, diagonal is diag, others
  // zero
  structure(0, 0) = BlockKind::Dense;
  for (isize i = 1; i < nblocks; ++i) {
    // first col/row
    structure(i, 0) = BlockKind::Dense;
    structure(0, i) = BlockKind::Dense;

    for (isize j = 1; j < nblocks; ++j) {
      structure(i, j) = BlockKind::Zero;
    }

    // diag
    structure(i, i) = BlockKind::Diag;
  }

  block_chol::BlockLDLT<Scalar> ldlt(tot_size, structure.copy());
  ldlt.findSparsifyingPermutation();
  ldlt.updateBlockPermutMatrix(structure);
  return ldlt;
}

} // namespace proxnlp
