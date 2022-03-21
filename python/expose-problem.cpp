#include "lienlp/python/fwd.hpp"
#include "lienlp/results.hpp"


namespace lienlp {
  namespace python {
    
    void exposeProblem()
    {
      bp::class_<context::Problem_t>("Problem",
                                     "Problem definition class.",
                                     bp::no_init);
    }

  } // namespace python
} // namespace lienlp

