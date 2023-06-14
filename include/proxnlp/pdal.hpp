/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/workspace.hpp"

namespace proxnlp {

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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using ConstraintObject = ConstraintObjectTpl<Scalar>;

  ALMeritFunctionTpl(const Problem &prob, const Scalar &beta);

  Scalar evaluate(const ConstVectorRef &x, const std::vector<VectorRef> &lams,
                  Workspace &workspace) const;

  void computeGradient(Workspace &workspace) const;

private:
  // fraction of mu to use in linesearch; reference to outer algorithm param
  const Scalar &beta_;
  const Problem &problem_;
};

} // namespace proxnlp

#include "proxnlp/pdal.hxx"

#ifdef PROXNLP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxnlp/pdal.txx"
#endif
