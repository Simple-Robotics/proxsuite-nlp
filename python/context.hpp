#pragma once

#include "lienlp/fwd.hpp"
#include "lienlp/macros.hpp"


namespace lienlp
{
namespace python
{

  namespace context
  {
    
    using Scalar = double;

    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using VectorXBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    using Problem_t = Problem<Scalar>;
    using Result_t = SResults<Scalar>;
    using Workspace_t = SWorkspace<Scalar>;
    using Cost_t = CostFunctionBase<Scalar>;
    using Constraint_t = ConstraintSetBase<Scalar>;
    using Functor_t = BaseFunctor<Scalar>;
    using DFunctor_t = DifferentiableFunctor<Scalar>;

    using ManifoldType = ManifoldAbstract<Scalar>;

    using VecFunc_t = void(const ConstVectorRef&, VectorRef) const;
    using VecFuncRet_t = VectorXs(const ConstVectorRef&) const;
    using MatFunc_t = void(const ConstVectorRef&, MatrixRef) const;
    using MatFuncRet_t = MatrixXs(const ConstVectorRef&) const;
    /// Signature of no-allocation vector-hessian product.
    using VHPFunc_t = void(const ConstVectorRef&, const ConstVectorRef&, MatrixRef) const;


  } // namespace context

} // namespace python
} // namespace lienlp


