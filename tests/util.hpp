#pragma once

#include "proxsuite-nlp/math.hpp"
#include "proxsuite-nlp/linalg/block-ldlt.hpp"
#include <EigenRand/EigenRand>

namespace {
using proxsuite::nlp::linalg::BlockKind;
using proxsuite::nlp::linalg::isize;
using proxsuite::nlp::linalg::SymbolicBlockMatrix;
using Scalar = double;
PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
} // namespace

/// Sample for the GOE (Gaussian Orthogonal Ensemble)
MatrixXs sampleGaussianOrthogonalEnsemble(Eigen::Index n);

/// Get a random, symmetric block-sparse matrix
MatrixXs getRandomSymmetricBlockMatrix(SymbolicBlockMatrix const &sym);
