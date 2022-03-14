
namespace lienlp {
  template<typename Scalar>
  void PDALFunction<Scalar>::gradient(
    const VectorXs& x,
    const VectorList& lams,
    const VectorList& lams_ext,
    VectorXs& out) const
  {
    VectorList lams_plus;
    lams_plus.reserve(m_prob->getNumConstraints());
    computePDALMultipliers(x, lams, lams_ext, lams_plus);
    m_lagr.gradient(x, lams_plus, out);
  }

  template<typename Scalar>
  void PDALFunction<Scalar>::hessian(
    const VectorXs& x,
    const VectorList& lams,
    const VectorList& lams_ext,
    MatrixXs& out) const
  {
    VectorList lams_plus;
    lams_plus.reserve(m_prob->getNumConstraints());
    computePDALMultipliers(x, lams, lams_ext, lams_plus);
    m_lagr.hessian(x, lams_plus, out);
  }

} // namespacelienlp