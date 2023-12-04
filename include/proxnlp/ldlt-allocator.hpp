/// @file
/// @brief     Utility function to allocate an LDLT solver for the Newton
/// iterations.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/linalg/block-ldlt.hpp"
#include "proxnlp/linalg/bunchkaufman.hpp"
#ifdef PROXNLP_ENABLE_PROXSUITE_LDLT
#include "proxnlp/linalg/proxsuite-ldlt-wrap.hpp"
#endif
#include <memory>

namespace proxnlp {

namespace {
using linalg::isize;
using linalg::SymbolicBlockMatrix;
using std::unique_ptr;
} // namespace

enum class LDLTChoice {
  /// Use our dense LDLT.
  DENSE,
  /// Use blocked LDLT.
  BLOCKSPARSE,
  /// Use Eigen's implementation.
  EIGEN,
  /// Use Proxsuite's LDLT.
  PROXSUITE
};

inline SymbolicBlockMatrix
create_default_block_structure(const std::vector<isize> &dims_primal,
                               const std::vector<isize> &dims_dual) {

  using linalg::BlockKind;

  isize nprim_blocks = (isize)dims_primal.size();
  isize ndual_blocks = (isize)dims_dual.size();
  isize nblocks = nprim_blocks + ndual_blocks;

  SymbolicBlockMatrix structure(nblocks, nblocks);
  isize *segment_lens = structure.segment_lens;

  for (unsigned int i = 0; i < nprim_blocks; ++i) {
    segment_lens[i] = dims_primal[i];
  }
  for (unsigned int i = 0; i < ndual_blocks; ++i) {
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

inline isize get_total_dim_helper(const std::vector<isize> &nprims,
                                  const std::vector<isize> &nduals) {
  return std::accumulate(nprims.begin(), nprims.end(), 0) +
         std::accumulate(nduals.begin(), nduals.end(), 0);
}

template <typename Scalar>
unique_ptr<linalg::ldlt_base<Scalar>>
allocate_ldlt_from_sizes(const std::vector<isize> &nprims,
                         const std::vector<isize> &nduals, LDLTChoice choice) {
  using ldlt_ptr_t = unique_ptr<linalg::ldlt_base<Scalar>>;
  const isize size = get_total_dim_helper(nprims, nduals);

  switch (choice) {
  case LDLTChoice::DENSE:
    return ldlt_ptr_t(new linalg::DenseLDLT<Scalar>(size));
  case LDLTChoice::BLOCKED: {
    SymbolicBlockMatrix structure =
        create_default_block_structure(nprims, nduals);

    auto *block_ldlt = new linalg::BlockLDLT<Scalar>(size, structure);
    block_ldlt->findSparsifyingPermutation();
    return ldlt_ptr_t(block_ldlt);
  }
  case LDLTChoice::EIGEN:
    return ldlt_ptr_t(new linalg::EigenLDLTWrapper<Scalar>(size));
  case LDLTChoice::PROXSUITE:
#ifdef PROXNLP_ENABLE_PROXSUITE_LDLT
    return ldlt_ptr_t(new linalg::ProxSuiteLDLTWrapper<Scalar>(size, size));
#else
    PROXNLP_RUNTIME_ERROR(
        "ProxSuite support is not enabled. You should recompile ProxNLP with "
        "the BUILD_WITH_PROXSUITE flag.");
#endif
  default:
    return nullptr;
  }
}

} // namespace proxnlp
