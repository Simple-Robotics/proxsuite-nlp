/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/math.hpp"
#include "proxnlp/macros.hpp"

#include <Eigen/Cholesky>

namespace proxnlp {

/// @brief    Specific linear algebra routines.
/// @details	Block-wise Cholesky and LDLT factorisation routines.
namespace linalg {

using isize = Eigen::Index;

/// @brief  Base interface for LDLT solvers.
template <typename Scalar> struct ldlt_base {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  virtual ldlt_base &compute(const MatrixRef &mat) = 0;
  virtual bool solveInPlace(MatrixRef b) const = 0;
  virtual Eigen::Diagonal<const MatrixXs> vectorD() const = 0;
  virtual const MatrixXs &matrixLDLT() const = 0;
  virtual MatrixXs reconstructedMatrix() const = 0;
  virtual Eigen::ComputationInfo info() const { return m_info; }
  virtual ~ldlt_base() = default;

protected:
  Eigen::ComputationInfo m_info;
};

template <typename Scalar> struct EigenLDLTWrapper : ldlt_base<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ldlt_base<Scalar>;

  EigenLDLTWrapper(isize size) : Base(), m_ldlt(size) {}
  EigenLDLTWrapper(const MatrixRef &mat) : m_ldlt(mat) {}
  EigenLDLTWrapper(const Eigen::LDLT<MatrixXs> &ldlt) : m_ldlt(ldlt) {}

  inline EigenLDLTWrapper &compute(const MatrixRef &mat) override {
    m_ldlt.compute(mat);
    return *this;
  }

  inline bool solveInPlace(MatrixRef b) const override {
    return m_ldlt.solveInPlace(b);
  }

  inline MatrixXs reconstructedMatrix() const override {
    return m_ldlt.reconstructedMatrix();
  }

  inline const MatrixXs &matrixLDLT() const override {
    return m_ldlt.matrixLDLT();
  }

  inline Eigen::ComputationInfo info() const override { return m_ldlt.info(); }

  inline Eigen::Diagonal<const MatrixXs> vectorD() const override {
    return m_ldlt.vectorD();
  }

protected:
  Eigen::LDLT<MatrixXs> m_ldlt;
};

} // namespace linalg
} // namespace proxnlp
