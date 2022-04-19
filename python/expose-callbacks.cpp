#include "lienlp/python/fwd.hpp"
#include "lienlp/helpers-base.hpp"
#include "lienlp/helpers/history-callback.hpp"

namespace lienlp
{
  namespace python
  {

    struct CallbackWrapper : helpers::base_callback<context::Scalar>,
                             bp::wrapper<helpers::base_callback<context::Scalar>>
    {
      CallbackWrapper() = default;
      void call(const context::Workspace& w, const context::Results& r)
      {
        this->get_override("call")(w, r);
      }
    };

    void exposeCallbacks()
    {
      using context::Scalar;
      using callback_t = helpers::base_callback<Scalar>;

      bp::register_ptr_to_python<shared_ptr<callback_t>>();

      bp::class_<CallbackWrapper, shared_ptr<CallbackWrapper>, boost::noncopyable>(
        "BaseCallback", "Base callback for solvers.", bp::init<>())
        .def("call", bp::pure_virtual(&CallbackWrapper::call), bp::args("self", "workspace", "results"))
        ;

      {
        using history_storage_t = decltype(helpers::history_callback<Scalar>::storage);

        bp::scope in_history = bp::class_<helpers::history_callback<Scalar>, bp::bases<callback_t>>(
          "HistoryCallback", "Store the history of solver's variables.",
          bp::init<bool, bool, bool>((
                      bp::arg("store_pd_vars") = true
                    , bp::arg("store_values") = true
                    , bp::arg("store_residuals") = true
                   ))
          )
          .def_readonly("storage", &helpers::history_callback<Scalar>::storage);

        bp::class_<history_storage_t, shared_ptr<history_storage_t>>("_history_storage")
          .def_readonly("xs", &history_storage_t::xs)
          .def_readonly("lams", &history_storage_t::lams)
          .def_readonly("values", &history_storage_t::values)
          .def_readonly("prim_infeas", &history_storage_t::prim_infeas)
          .def_readonly("dual_infeas", &history_storage_t::dual_infeas)
          .def_readonly("ls_alphas", &history_storage_t::ls_alphas)
          .def_readonly("ls_values", &history_storage_t::ls_values)
          .def_readonly("d1_s", &history_storage_t::d1_s)
          ;
      }
    }    
  } // namespace python
} // namespace lienlp

