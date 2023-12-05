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
#include <boost/variant.hpp>

namespace proxnlp {

namespace {
using linalg::isize;
} // namespace

enum class LDLTChoice {
  /// Use our dense LDLT.
  DENSE,
  /// Use Bunch-Kaufman factorization
  BUNCHKAUFMAN,
  /// Use blocked LDLT.
  BLOCKSPARSE,
  /// Use Eigen's implementation.
  EIGEN,
  /// Use Proxsuite's LDLT.
  PROXSUITE
};

template <typename Scalar,
          class MatrixType = typename math_types<Scalar>::MatrixXs>
using LDLTVariant =
    boost::variant<linalg::DenseLDLT<Scalar>, linalg::BlockLDLT<Scalar>,
                   Eigen::LDLT<MatrixType>, Eigen::BunchKaufman<MatrixType>
#ifdef PROXNLP_ENABLE_PROXSUITE_LDLT
                   ,
                   linalg::ProxSuiteLDLTWrapper<Scalar>
#endif
                   >;

inline linalg::SymbolicBlockMatrix
create_default_block_structure(const std::vector<isize> &dims_primal,
                               const std::vector<isize> &dims_dual) {

  using linalg::BlockKind;

  isize nprim_blocks = (isize)dims_primal.size();
  isize ndual_blocks = (isize)dims_dual.size();
  isize nblocks = nprim_blocks + ndual_blocks;

  linalg::SymbolicBlockMatrix structure(nblocks, nblocks);
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
LDLTVariant<Scalar> allocate_ldlt_from_sizes(const std::vector<isize> &nprims,
                                             const std::vector<isize> &nduals,
                                             LDLTChoice choice) {
  const isize size = get_total_dim_helper(nprims, nduals);
  using MatrixXs = typename math_types<Scalar>::MatrixXs;

  switch (choice) {
  case LDLTChoice::DENSE:
    return linalg::DenseLDLT<Scalar>(size);
  case LDLTChoice::BUNCHKAUFMAN:
    return Eigen::BunchKaufman<MatrixXs>(size);
  case LDLTChoice::BLOCKSPARSE: {
    auto structure = create_default_block_structure(nprims, nduals);

    linalg::BlockLDLT<Scalar> block_ldlt(size, structure);
    block_ldlt.findSparsifyingPermutation();
    return block_ldlt;
  }
  case LDLTChoice::EIGEN:
    return Eigen::LDLT<MatrixXs>(size);
  case LDLTChoice::PROXSUITE:
#ifdef PROXNLP_ENABLE_PROXSUITE_LDLT
    return linalg::ProxSuiteLDLTWrapper<Scalar>(size, size);
#else
    PROXNLP_RUNTIME_ERROR(
        "ProxSuite support is not enabled. You should recompile ProxNLP with "
        "the BUILD_WITH_PROXSUITE flag.");
#endif
  }
}

namespace internal {

/// Compute signature of matrix from Bunch-Kaufman factorization.
template <typename MatrixType, typename Signature, int UpLo>
auto bunch_kaufman_compute_signature(
    Eigen::BunchKaufman<MatrixType, UpLo> const &factor, Signature &signature) {
  // TODO: finish implementing this
  PROXNLP_RUNTIME_ERROR("Not implemented yet.");
}

} // namespace internal

struct ComputeSignatureVisitor {
  template <typename Fac> void operator()(const Fac &facto) const {
    auto sign = facto.vectorD().cwiseSign();
    signature = sign.template cast<int>();
  }

  template <typename MatType>
  void operator()(const Eigen::BunchKaufman<MatType> &facto) const {
    internal::bunch_kaufman_compute_signature(facto, signature);
  }
  Eigen::VectorXi &signature;
};

} // namespace proxnlp
