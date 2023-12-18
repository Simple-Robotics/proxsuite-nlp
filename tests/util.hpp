#pragma once

#include "proxnlp/math.hpp"
#include "proxnlp/linalg/block-ldlt.hpp"
#include <EigenRand/EigenRand>

namespace {
using proxnlp::linalg::BlockKind;
using proxnlp::linalg::isize;
using proxnlp::linalg::SymbolicBlockMatrix;
using Scalar = double;
PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
} // namespace

/// Sample for the GOE (Gaussian Orthogonal Ensemble)
MatrixXs sampleGaussianOrhogonalEnsemble(Eigen::Index n);

/// Get a random, symmetric block-sparse matrix
MatrixXs getRandomSymmetricBlockMatrix(SymbolicBlockMatrix const &sym);
