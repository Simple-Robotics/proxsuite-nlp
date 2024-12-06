/// @file
/// @brief     Utility function to allocate an LDLT solver for the Newton
/// iterations.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/linalg/block-ldlt.hpp"
#include "proxsuite-nlp/linalg/bunchkaufman.hpp"
#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT
#include "proxsuite-nlp/linalg/proxsuite-ldlt-wrap.hpp"
#endif
#include <boost/variant.hpp>
#include <array>

namespace proxsuite {
namespace nlp {

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
                   Eigen::LDLT<MatrixType>, BunchKaufman<MatrixType>
#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT
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
    return BunchKaufman<MatrixXs>(size);
  case LDLTChoice::BLOCKSPARSE: {
    auto structure = create_default_block_structure(nprims, nduals);

    linalg::BlockLDLT<Scalar> block_ldlt(size, structure);
    block_ldlt.findSparsifyingPermutation();
    return block_ldlt;
  }
  case LDLTChoice::EIGEN:
    return Eigen::LDLT<MatrixXs>(size);
  case LDLTChoice::PROXSUITE:
#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT
    return linalg::ProxSuiteLDLTWrapper<Scalar>(size, size);
#else
  default:
    PROXSUITE_NLP_RUNTIME_ERROR(
        "ProxSuite support is not enabled. You should recompile ProxNLP with "
        "the BUILD_WITH_PROXSUITE_SUPPORT flag.");
#endif
  }
}

namespace internal {

/// Compute signature of matrix from Bunch-Kaufman factorization.
template <typename MatrixType, typename Signature, int UpLo>
void bunch_kaufman_compute_signature(
    BunchKaufman<MatrixType, UpLo> const &factor, Signature &signature) {
  using Eigen::Index;
  using Scalar = typename MatrixType::Scalar;
  using Real = typename Eigen::NumTraits<Scalar>::Real;

  Index n = factor.rows();
  signature.conservativeResize(n);

  Index k = 0;
  const MatrixType &a = factor.matrixLDLT();

  while (k < n) {
    Index p = factor.pivots()[k];
    if (p < 0) {
      // 2x2 block
      Real ak = Eigen::numext::real(a(k, k));
      Real akp1 = Eigen::numext::real(a(k + 1, k + 1));
      Scalar akp1k = factor.subdiag()[k];
      Real tr = ak + akp1;
      Real det = ak * akp1 - Eigen::numext::abs2(akp1k);

      if (std::abs(det) <= std::numeric_limits<Scalar>::epsilon()) {
        signature[k] = 0;
        signature[k + 1] = int(math::sign(tr));
      } else if (det > Scalar(0)) {
        signature[k] = int(math::sign(tr));
        signature[k + 1] = signature[k];
      } else {
        // det < 0
        signature[k] = -1;
        signature[k + 1] = +1;
      }

      k += 2;
    } else {
      Real ak = Eigen::numext::real(a(k, k));
      signature[k] = int(math::sign(ak));
      k += 1;
    }
  }
}

} // namespace internal

struct ComputeSignatureVisitor {
  template <typename Fac> void operator()(const Fac &facto) const {
    auto sign = facto.vectorD().cwiseSign();
    signature = sign.template cast<int>();
  }

  template <typename MatType>
  void operator()(const BunchKaufman<MatType> &facto) const {
    internal::bunch_kaufman_compute_signature(facto, signature);
  }
  Eigen::VectorXi &signature;
};

inline std::array<int, 3>
computeInertiaTuple(const Eigen::Ref<Eigen::VectorXi const> &signature) {
  using Eigen::Index;
  Index n = signature.size();
  int np = 0;
  int n0 = 0;
  int nn = 0;
  for (Index i = 0; i < n; i++) {
    switch (signature(i)) {
    case 1:
      np++;
      break;
    case 0:
      n0++;
      break;
    case -1:
      nn++;
      break;
    default:
      PROXSUITE_NLP_RUNTIME_ERROR(
          "Signature vector should only contain values O, 1, -1.");
      break;
    }
  }
  return {np, nn, n0};
}

} // namespace nlp
} // namespace proxsuite
