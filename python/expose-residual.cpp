#include "lienlp/python/fwd.hpp"

#include "lienlp/modelling/residuals/linear.hpp"

#include <boost/python/overloads.hpp>


namespace lienlp
{
namespace python
{
  namespace bp = boost::python;

  void exposeResiduals()
  {
    using context::VectorXs;
    using context::MatrixXs;
    using context::DFunctor_t;

    bp::class_<LinearResidual<context::Scalar>, bp::bases<DFunctor_t>>(
      "LinearResidual",
      bp::init<MatrixXs, VectorXs>(bp::args("A", "b")));
  }

} // namespace python
} // namespace lienlp
