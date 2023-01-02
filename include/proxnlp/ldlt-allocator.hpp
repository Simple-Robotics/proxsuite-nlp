/// @file
/// @brief     Utility function to allocate an LDLT solver for the Newton
/// iterations.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "linalg/blocks.hpp"
#include "proxnlp/problem-base.hpp"

namespace proxnlp {

namespace {
using linalg::BlockKind;
using linalg::BlockLDLT;
using linalg::isize;
using linalg::SymbolicBlockMatrix;
} // namespace

enum class LDLTChoice {
  /// Use our dense LDLT.
  DENSE,
  /// Use blocked LDLT.
  BLOCKED,
  /// Use Eigen's implementation.
  EIGEN,
};

template <typename Scalar>
BlockLDLT<Scalar> *
allocate_block_ldlt_from_structure(long nprim,
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

  BlockLDLT<Scalar> *ldlt = new BlockLDLT<Scalar>(tot_size, structure.copy());
  (*ldlt).findSparsifyingPermutation().updateBlockPermutationMatrix(structure);
  return ldlt;
}

template <typename Scalar>
unique_ptr<linalg::ldlt_base<Scalar>>
allocate_ldlt_from_sizes(const isize nprim, const std::vector<isize> &nduals,
                         LDLTChoice choice) {
  using ldlt_ptr_t = unique_ptr<linalg::ldlt_base<Scalar>>;
  const long size =
      nprim + std::accumulate(nduals.begin(), nduals.end(), isize(0));
  switch (choice) {
  case LDLTChoice::DENSE:
    return ldlt_ptr_t(new linalg::DenseLDLT<Scalar>(size));
  case LDLTChoice::BLOCKED: {
    return ldlt_ptr_t(
        allocate_block_ldlt_from_structure<Scalar>(nprim, nduals));
  }
  case LDLTChoice::EIGEN:
    return ldlt_ptr_t(new linalg::EigenLDLTWrapper<Scalar>(size));
  default:
    return nullptr;
  }
}

template <typename Scalar>
unique_ptr<linalg::ldlt_base<Scalar>>
allocate_ldlt_from_problem(const ProblemTpl<Scalar> &prob, LDLTChoice choice) {
  std::vector<isize> nduals(prob.getNumConstraints());
  for (std::size_t i = 0; i < nduals.size(); ++i)
    nduals[i] = prob.getConstraintDim(i);
  return allocate_ldlt_from_sizes<Scalar>(prob.ndx(), nduals, choice);
}

} // namespace proxnlp
