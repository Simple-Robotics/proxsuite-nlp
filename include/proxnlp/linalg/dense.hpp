/// @file
/// @author Sarah El-Kazdadi
/// @author Wilson Jallet
/// @brief Routines for Cholesky factorization.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/linalg/ldlt-base.hpp"

namespace proxnlp {
namespace linalg {

using Eigen::internal::LDLT_Traits;

namespace backend {

/// At the end of the execution, @param a contains
/// the lower-triangular matrix \f$L\f$ in the LDLT decomposition.
/// More precisely: a stores L -sans its diagonal which is all ones.
/// The diagonal of @param a contains the diagonal matrix @f$D@f$.
template <typename Derived>
inline bool ldlt_in_place_unblocked(Eigen::MatrixBase<Derived> &a) {
  const isize n = a.rows();
  if (n <= 1) {
    return true;
  }

  isize j = 0;
  while (true) {
    auto l10 = a.row(j).head(j);
    auto d0 = a.diagonal().head(j);
    auto work = a.col(n - 1).head(j);

    work = l10.transpose().cwiseProduct(d0);
    a(j, j) -= work.dot(l10);

    if (j + 1 == n) {
      return true;
    }

    const isize rem = n - j - 1;

    auto l20 = a.bottomLeftCorner(rem, j);
    auto l21 = a.col(j).tail(rem);

    l21.noalias() -= l20 * work;
    l21 *= 1 / a(j, j);
    ++j;
  }
}

static constexpr isize UNBLK_THRESHOLD = 128;

/// A recursive, in-place implementation of the LDLT decomposition.
/// To be applied to dense blocks.
template <typename Derived>
inline bool dense_ldlt_in_place(Eigen::MatrixBase<Derived> &a) {
  using PlainObject = typename Derived::PlainObject;
  using MatrixRef = Eigen::Ref<PlainObject>;
  const isize n = a.rows();
  if (n <= UNBLK_THRESHOLD) {
    return backend::ldlt_in_place_unblocked(a);
  } else {
    const isize bs = (n + 1) / 2;
    const isize rem = n - bs;

    MatrixRef l00 = a.block(0, 0, bs, bs);
    Eigen::Block<Derived> l10 = a.block(bs, 0, rem, bs);
    MatrixRef l11 = a.block(bs, bs, rem, rem);

    backend::dense_ldlt_in_place(l00);
    auto d0 = l00.diagonal();

    l00.transpose()
        .template triangularView<Eigen::UnitUpper>()
        .template solveInPlace<Eigen::OnTheRight>(l10);

    auto work = a.block(0, bs, bs, rem).transpose();
    work = l10;
    l10 = l10 * d0.asDiagonal().inverse();

    l11.template triangularView<Eigen::Lower>() -= l10 * work.transpose();

    return backend::dense_ldlt_in_place(l11);
  }
}

/// Taking the decomposed LDLT matrix @param mat, solve the original linear
/// system.
template <typename MatDerived, typename Rhs>
inline bool dense_ldlt_solve_in_place(MatDerived &mat, Rhs &b) {
  using Scalar = typename MatDerived::Scalar;
  using Traits = LDLT_Traits<MatDerived, Eigen::Lower>;
  Traits::getL(mat).solveInPlace(b);

  using std::abs;
  auto vecD(mat.diagonal());
  const Scalar tol = std::numeric_limits<Scalar>::min();
  for (isize i = 0; i < vecD.size(); ++i) {
    if (abs(vecD(i)) > tol)
      b.row(i) /= vecD(i);
    else
      b.row(i).setZero();
  }

  Traits::getU(mat).solveInPlace(b);
  return true;
}

template <typename Scalar>
inline void
dense_ldlt_reconstruct(typename math_types<Scalar>::ConstMatrixRef const &mat,
                       typename math_types<Scalar>::MatrixRef res) {
  using ConstMatrixRef = typename math_types<Scalar>::ConstMatrixRef;
  using Traits = LDLT_Traits<ConstMatrixRef, Eigen::Lower>;
  res = Traits::getU(mat) * res;

  auto vecD = mat.diagonal();
  res = vecD.asDiagonal() * res;

  res = Traits::getL(mat) * res;
}
} // namespace backend

/// @brief  A fast, recursive divide-and-conquer LDLT algorithm.
template <typename Scalar> struct DenseLDLT : ldlt_base<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ldlt_base<Scalar>;

  DenseLDLT() = default;
  explicit DenseLDLT(isize size) : Base(), m_matrix(size, size) {
    m_matrix.setZero();
  }
  explicit DenseLDLT(MatrixRef a) : Base(), m_matrix(a) {
    m_info = backend::dense_ldlt_in_place(m_matrix) ? Eigen::Success
                                                    : Eigen::NumericalIssue;
  }

  DenseLDLT &compute(const MatrixRef &mat) override {
    m_matrix = mat;
    m_info = backend::dense_ldlt_in_place(m_matrix) ? Eigen::Success
                                                    : Eigen::NumericalIssue;
    return *this;
  }

  const MatrixXs &matrixLDLT() const override { return m_matrix; }

  bool solveInPlace(MatrixRef b) const override {
    return backend::dense_ldlt_solve_in_place(m_matrix, b);
  }

  MatrixXs reconstructedMatrix() const override {
    MatrixXs res(m_matrix.rows(), m_matrix.cols());
    res.setIdentity();
    backend::dense_ldlt_reconstruct<Scalar>(m_matrix, res);
    return res;
  }

  inline Eigen::Diagonal<const MatrixXs> vectorD() const override {
    return m_matrix.diagonal();
  }

protected:
  MatrixXs m_matrix;
  using Base::m_info;
};

} // namespace linalg
} // namespace proxnlp
