#pragma once

#include "proxnlp/meritfuncs/pdal.hpp"


namespace proxnlp
{
  template<typename Scalar>
  Scalar PDALFunction<Scalar>::operator()(
    const ConstVectorRef& x,
    const VectorOfRef& lams,
    const VectorOfRef& lams_ext) const
  {
    Scalar result_ = m_prob->m_cost.call(x);

    const std::size_t nc = m_prob->getNumConstraints();
    for (std::size_t i = 0; i < nc; i++)
    {
      const auto cstr = m_prob->getConstraint(i);
      VectorXs cval = cstr->normalConeProjection(cstr->m_func(x) + m_mu * lams_ext[i]);
      result_ += Scalar(0.5) * (m_muInv * cval.squaredNorm() - m_mu * lams_ext[i].squaredNorm());
      result_ += Scalar(0.5) * m_muInv * (cval - m_mu * lams[i]).squaredNorm();
    }

    return result_;
  }

  template<typename Scalar>
  void PDALFunction<Scalar>::computeGradient(
    const ConstVectorRef& x,
    const VectorOfRef& lams,
    const VectorOfRef& lams_ext,
    VectorRef out) const
  {
    VectorOfRef lams_plus;
    VectorXs data(m_prob->getTotalConstraintDim());
    helpers::allocateMultipliersOrResiduals(*m_prob, data, lams_plus);
    computePDALMultipliers(x, lams, lams_ext, lams_plus);
    m_lagr.computeGradient(x, lams_plus, out);
  }

  template<typename Scalar>
  void PDALFunction<Scalar>::computeHessian(
    const ConstVectorRef& x,
    const VectorOfRef& lams,
    const VectorOfRef& lams_ext,
    MatrixRef out) const
  {
    /// Compute cost hessian // TODO rip this out, use workspace
    m_prob->m_cost.computeHessian(x, out);
    const std::size_t nc = m_prob->getNumConstraints();
    const int ndx = m_prob->m_cost.ndx();

    VectorOfRef lams_plus;
    VectorXs data(m_prob->getTotalConstraintDim());
    helpers::allocateMultipliersOrResiduals(*m_prob, data, lams_plus);
    computePDALMultipliers(x, lams, lams_ext, lams_plus);

    // m_lagr.computeHessian(x, lams_plus, out);  // useless because recompute

    MatrixXs J, vhp_buffer(ndx, ndx); // TODO refactor this allocation using workspace
    for (std::size_t i = 0; i < nc; i++)
    {
      typename Problem::ConstraintPtr cstr = m_prob->getConstraint(i);
      J.resize(cstr->nr(), ndx);
      cstr->m_func.computeJacobian(x, J);
      cstr->m_func.vectorHessianProduct(x, lams_plus[i], vhp_buffer);
      out.noalias() += vhp_buffer + 2 * m_mu * J.transpose() * J;
    }
  }

  template<typename Scalar>
  void PDALFunction<Scalar>::computeFirstOrderMultipliers(
    const ConstVectorRef& x,
    const VectorOfRef& lams_ext,
    VectorOfRef& out) const
  {
    for (std::size_t i = 0; i < m_prob->getNumConstraints(); i++)
    {
      auto cstr = m_prob->getConstraint(i);
      out[i].noalias() = cstr->normalConeProjection(lams_ext[i] + m_muInv * cstr->m_func(x));
    }
  }

  template<typename Scalar>
  void PDALFunction<Scalar>::computePDALMultipliers(
    const ConstVectorRef& x,
    const VectorOfRef& lams,
    const VectorOfRef& lams_ext,
    VectorOfRef& out) const
  {
    // TODO fix calling this again; grab values from workspace
    computeFirstOrderMultipliers(x, lams_ext, out);
    for (std::size_t i = 0; i < m_prob->getNumConstraints(); i++)
    {
      auto cstr = m_prob->getConstraint(i);
      out[i].noalias() = cstr->normalConeProjection(2 * out[i] - lams[i]);
    }
  }
} // namespace proxnlp