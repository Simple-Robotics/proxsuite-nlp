
namespace lienlp {

  template<typename Scalar>
  void PDALFunction<Scalar>::computeGradient(
    const ConstVectorRef& x,
    const VectorOfVectors& lams,
    const VectorOfVectors& lams_ext,
    VectorRef out) const
  {
    VectorOfVectors lams_plus;
    lams_plus.reserve(m_prob->getNumConstraints());
    computePDALMultipliers(x, lams, lams_ext, lams_plus);
    m_lagr.computeGradient(x, lams_plus, out);
  }

  template<typename Scalar>
  void PDALFunction<Scalar>::computeHessian(
    const ConstVectorRef& x,
    const VectorOfVectors& lams,
    const VectorOfVectors& lams_ext,
    MatrixRef out) const
  {
    /// Compute cost hessian // TODO rip this out, use workspace
    m_prob->m_cost.computeHessian(x, out);
    const auto num_c = m_prob->getNumConstraints();
    const auto ndx = m_prob->m_cost.ndx();

    // Compute appropriate multiplier estimates, TODO rip this out
    VectorOfVectors lams_plus;
    lams_plus.reserve(num_c);
    computePDALMultipliers(x, lams, lams_ext, lams_plus);

    // m_lagr.computeHessian(x, lams_plus, out);  // useless because recompute

    MatrixXs J, vhp_buffer(ndx, ndx); // TODO refactor this allocation using workspace
    for (std::size_t i = 0; i < num_c; i++)
    {
      typename Prob_t::CstrPtr cstr = m_prob->getCstr(i);
      J.resize(cstr->nr(), ndx);
      cstr->m_func.computeJacobian(x, J);
      cstr->m_func.vectorHessianProduct(x, lams_plus[i], vectorHessianProductBuf);
      out.noalias() += vhp_buffer + 2 * m_muEq * J.transpose() * J;
    }
  }

} // namespace lienlp