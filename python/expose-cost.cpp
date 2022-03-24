
#include "lienlp/python/fwd.hpp"
#include "lienlp/cost-function.hpp"


namespace lienlp {
namespace python {

  void exposeCost()
  {
    bp::class_<context::Cost_t, boost::noncopyable>(
      "CostFunctionBase", bp::no_init);
  }  

} // namespace python
} // namespace lienlp

