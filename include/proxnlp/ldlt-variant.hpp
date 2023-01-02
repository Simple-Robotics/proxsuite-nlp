/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "linalg/blocks.hpp"

#include <boost/variant.hpp>

namespace proxnlp {

using boost::variant;

/// @brief A variant type for all LDLT implementations to be made available to
/// the user.
template <typename Scalar>
using LDLT_variant =
    variant<linalg::DenseLDLT<Scalar>, linalg::BlockLDLT<Scalar>>;

namespace linalg {

template <typename Scalar>
BlockLDLT<Scalar>
initialize_block_ldlt_from_structure(long nprim,
                                     const std::vector<long> &nduals) {
  const isize nblocks = 1 + (isize)nduals.size();

  isize tot_size =
      nprim + std::accumulate(nduals.begin(), nduals.end(), isize(0));

  std::vector<BlockKind> blocks((std::size_t)(nblocks * nblocks));
  std::vector<long> seg_lens = nduals;
  seg_lens.insert(seg_lens.begin(), nprim);

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

  BlockLDLT<Scalar> ldlt(tot_size, structure.copy());
  ldlt.findSparsifyingPermutation();
  ldlt.updateBlockPermutationMatrix(structure);
  return ldlt;
}

} // namespace linalg
} // namespace proxnlp
