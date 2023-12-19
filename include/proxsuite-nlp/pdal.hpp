/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/fwd.hpp"
#include "proxsuite-nlp/workspace.hpp"

namespace proxsuite {
namespace nlp {

///
/// @brief   Primal-dual augmented Lagrangian-type merit function.
///
/// Primal-dual Augmented Lagrangian function, extending
/// the function from Gill & Robinson (2012) to inequality constraints.
/// For inequality constraints of the form \f$ c(x) \in \calC \f$ and an
/// objective function \f$ f\colon\calX \to \RR \f$,
/// \f[
///  \calM_{\mu}(x, \lambda; \lambda_e) = f(x) + \frac{1}{2\mu}
/// \dist(\proj_\calC(c(x) + \mu (\lambda_e - \lambda/2))^2 +
/// \frac{\mu}{4}\|\lambda\|^2. \f]
///
template <typename _Scalar> struct ALMeritFunctionTpl {
public:
  using Scalar = _Scalar;
  PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using ConstraintObject = ConstraintObjectTpl<Scalar>;

  ALMeritFunctionTpl(const Problem &prob, const Scalar &beta);

  Scalar evaluate(const ConstVectorRef &x, const std::vector<VectorRef> &lams,
                  Workspace &workspace) const;

  void computeGradient(const std::vector<VectorRef> &lams,
                       Workspace &workspace) const;

private:
  // fraction of mu to use in linesearch; reference to outer algorithm param
  const Scalar &beta_;
  const Problem &problem_;
};

} // namespace nlp
} // namespace proxsuite

#include "proxsuite-nlp/pdal.hxx"

#ifdef PROXSUITE_NLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxsuite-nlp/pdal.txx"
#endif
