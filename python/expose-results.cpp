#include "lienlp/python/fwd.hpp"
#include "lienlp/results.hpp"


namespace lienlp
{
namespace python
{

  void exposeResults()
  {
    bp::class_<context::Result_t>(
      "Results", "Results holder struct.", bp::no_init);
  }  

} // namespace python
} // namespace lienlp

