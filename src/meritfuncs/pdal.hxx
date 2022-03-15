
namespace lienlp {

  template<typename Scalar>
  void PDALFunction<Scalar>::computeGradient(
    const ConstVectorRef& x,
    const VectorOfVectors& lams,
    const VectorOfVectors& lams_ext,
    RefVector out) const
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
    RefMatrix out) const
  {
    VectorOfVectors lams_plus;
    lams_plus.reserve(m_prob->getNumConstraints());
    computePDALMultipliers(x, lams, lams_ext, lams_plus);
    m_lagr.computeHessian(x, lams_plus, out);
  }

} // namespacelienlp