
namespace lienlp {
  template<typename Scalar>
  void PDALFunction<Scalar>::gradient(
    const ConstVectorRef& x,
    const VectorOfVectors& lams,
    const VectorOfVectors& lams_ext,
    RefVector out) const
  {
    VectorOfVectors lams_plus;
    lams_plus.reserve(m_prob->getNumConstraints());
    computePDALMultipliers(x, lams, lams_ext, lams_plus);
    m_lagr.gradient(x, lams_plus, out);
  }

  template<typename Scalar>
  void PDALFunction<Scalar>::hessian(
    const ConstVectorRef& x,
    const VectorOfVectors& lams,
    const VectorOfVectors& lams_ext,
    RefMatrix out) const
  {
    VectorOfVectors lams_plus;
    lams_plus.reserve(m_prob->getNumConstraints());
    computePDALMultipliers(x, lams, lams_ext, lams_plus);
    m_lagr.hessian(x, lams_plus, out);
  }

} // namespacelienlp