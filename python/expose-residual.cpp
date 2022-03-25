#include "lienlp/python/fwd.hpp"

#include "lienlp/modelling/residuals/linear.hpp"

#include <boost/python/overloads.hpp>


namespace lienlp
{
namespace python
{
  namespace bp = boost::python;
  using context::DFunctor_t;
  struct ResidualWrap : DFunctor_t, bp::wrapper<DFunctor_t>
  {
    void computeJacobian(
      const ConstVectorRef& x,
      Eigen::Ref<JacobianType> Jout) const
    {
      this->get_override("computeJacobian")(x, Jout);
    }
  };

  void exposeResiduals()
  {
    using context::VectorXs;
    using context::MatrixXs;

    bp::class_<LinearResidual<context::Scalar>, bp::bases<DFunctor_t>>(
      "LinearResidual",
      bp::init<MatrixXs, VectorXs>());
  }

} // namespace python
} // namespace lienlp
