#pragma once

#include "proxnlp/math.hpp"
#include "proxnlp/linalg/block-ldlt.hpp"
#include <EigenRand/EigenRand>

namespace {
using proxnlp::linalg::BlockKind;
using proxnlp::linalg::isize;
using proxnlp::linalg::SymbolicBlockMatrix;
using Scalar = double;
PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
} // namespace

/// Sample for the GOE (Gaussian Orthogonal Ensemble)
MatrixXs sampleGaussianOrhogonalEnsemble(Eigen::Index n);

/// Sample from the Wishart distribution
MatrixXs sampleWishart(Eigen::Index dim, Eigen::Index m);

/// Get a random, symmetric block-sparse matrix
MatrixXs getRandomSymmetricBlockMatrix(SymbolicBlockMatrix const &sym);
