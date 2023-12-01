/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/math.hpp"
#include "proxnlp/exceptions.hpp"
#include "proxnlp/macros.hpp"

#include <Eigen/Cholesky>

namespace proxnlp {

/// @brief    Specific linear algebra routines.
/// @details	Block-wise Cholesky and LDLT factorisation routines.
namespace linalg {

using isize = Eigen::Index;
using Eigen::internal::SignMatrix;

/// @brief  Base interface for LDLT solvers.
template <typename Scalar> struct ldlt_base {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using DView = Eigen::Map<const VectorXs, Eigen::Unaligned,
                           Eigen::InnerStride<Eigen::Dynamic>>;

  template <typename Mat> static DView diag_view_impl(Mat &&mat) {
    Eigen::InnerStride<Eigen::Dynamic> stride{mat.outerStride() + 1};
    return {mat.data(), mat.rows(), 1, stride};
  }

  virtual ldlt_base &compute(const ConstMatrixRef &mat) = 0;
  virtual bool solveInPlace(MatrixRef b) const = 0;
  virtual DView vectorD() const = 0;
  virtual const MatrixXs &matrixLDLT() const {
    PROXNLP_RUNTIME_ERROR("Not implemented");
  }
  virtual MatrixXs reconstructedMatrix() const = 0;
  virtual Eigen::ComputationInfo info() const { return m_info; }
  SignMatrix sign() const { return m_sign; }
  virtual ~ldlt_base() = default;

protected:
  Eigen::ComputationInfo m_info;
  SignMatrix m_sign = SignMatrix::ZeroSign;
};

template <typename Scalar> struct EigenLDLTWrapper : ldlt_base<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ldlt_base<Scalar>;
  using DView = typename Base::DView;

  EigenLDLTWrapper(isize size) : Base(), m_ldlt(size) {}
  EigenLDLTWrapper(const MatrixRef &mat) : m_ldlt(mat) {}
  EigenLDLTWrapper(const Eigen::LDLT<MatrixXs> &ldlt) : m_ldlt(ldlt) {}

  inline EigenLDLTWrapper &compute(const ConstMatrixRef &mat) override {
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

  inline DView vectorD() const override {
    return Base::diag_view_impl(m_ldlt.matrixLDLT());
  }

protected:
  Eigen::LDLT<MatrixXs> m_ldlt;
};

} // namespace linalg
} // namespace proxnlp
