/// @file
/// @brief     Utility function to allocate an LDLT solver for the Newton
/// iterations.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "linalg/blocks.hpp"
#include <memory>

namespace proxnlp {

namespace {
using linalg::BlockLDLT;
using linalg::isize;
using linalg::SymbolicBlockMatrix;
using std::unique_ptr;
} // namespace

enum class LDLTChoice {
  /// Use our dense LDLT.
  DENSE,
  /// Use blocked LDLT.
  BLOCKED,
  /// Use Eigen's implementation.
  EIGEN,
};

SymbolicBlockMatrix
create_default_block_structure(const std::vector<isize> &dims_primal,
                               const std::vector<isize> &dims_dual);

template <typename Scalar>
BlockLDLT<Scalar> *
allocate_block_ldlt_from_structure(const std::vector<isize> &nprims,
                                   const std::vector<isize> &nduals) {
  SymbolicBlockMatrix structure =
      create_default_block_structure(nprims, nduals);

  isize tot_size = std::accumulate(nprims.begin(), nprims.end(), isize(0)) +
                   std::accumulate(nduals.begin(), nduals.end(), isize(0));

  return new BlockLDLT<Scalar>(tot_size, structure);
}

template <typename Scalar>
unique_ptr<linalg::ldlt_base<Scalar>>
allocate_ldlt_from_sizes(const std::vector<isize> &nprims,
                         const std::vector<isize> &nduals, LDLTChoice choice) {
  using ldlt_ptr_t = unique_ptr<linalg::ldlt_base<Scalar>>;
  const long size = std::accumulate(nprims.begin(), nprims.end(), isize(0)) +
                    std::accumulate(nduals.begin(), nduals.end(), isize(0));
  switch (choice) {
  case LDLTChoice::DENSE:
    return ldlt_ptr_t(new linalg::DenseLDLT<Scalar>(size));
  case LDLTChoice::BLOCKED: {
    BlockLDLT<Scalar> *block_ldlt =
        allocate_block_ldlt_from_structure<Scalar>(nprims, nduals);
    auto structure = block_ldlt->structure().copy();
    block_ldlt->findSparsifyingPermutation();
    block_ldlt->updateBlockPermutationMatrix(structure);
    return ldlt_ptr_t(block_ldlt);
  }
  case LDLTChoice::EIGEN:
    return ldlt_ptr_t(new linalg::EigenLDLTWrapper<Scalar>(size));
  default:
    return nullptr;
  }
}

} // namespace proxnlp
