#include "proxnlp/modelling/spaces/cartesian-product.hpp"

#include "proxnlp/python/fwd.hpp"

namespace proxnlp::python {

using context::Manifold;
using context::Scalar;
using ManifoldPtr = shared_ptr<Manifold>;
using context::ConstVectorRef;
using context::VectorRef;
using context::VectorXs;
using CartesianProduct = CartesianProductTpl<Scalar>;

std::vector<VectorXs> copy_vec_constref(const std::vector<ConstVectorRef> &x) {
  std::vector<VectorXs> out;
  for (const auto &c : x)
    out.push_back(c);
  return out;
}

void exposeCartesianProduct() {
  const std::string split_doc =
      "Takes an point on the product manifold and splits it up between the two "
      "base manifolds.";
  const std::string split_vec_doc =
      "Takes a tangent vector on the product manifold and splits it up.";

  using MutSplitSig =
      std::vector<VectorRef> (CartesianProduct::*)(VectorRef) const;

  bp::class_<CartesianProduct, bp::bases<Manifold>>(
      "CartesianProduct", "Cartesian product of two or more manifolds.",
      bp::init<const std::vector<ManifoldPtr> &>(bp::args("self", "spaces")))
      .def(
          bp::init<ManifoldPtr, ManifoldPtr>(bp::args("self", "left", "right")))
      .def("getComponent", &CartesianProduct::getComponent,
           bp::return_internal_reference<>(), bp::args("self", "i"),
           "Get the i-th component of the Cartesian product.")
      .def(
          "addComponent",
          +[](CartesianProduct &m, ManifoldPtr const &p) { m.addComponent(p); },
          bp::args("self", "c"), "Add a component to the Cartesian product.")
      .def(
          "addComponent",
          +[](CartesianProduct &m, shared_ptr<CartesianProduct> const &p) {
            m.addComponent(p);
          },
          bp::args("self", "c"), "Add a component to the Cartesian product.")
      .add_property("num_components", &CartesianProduct::numComponents,
                    "Get the number of components in the Cartesian product.")
      .def(
          "split",
          +[](CartesianProduct const &m, const ConstVectorRef &x) {
            return copy_vec_constref(m.split(x));
          },
          bp::args("self", "x"), split_doc.c_str())
      .def<MutSplitSig>(
          "split", &CartesianProduct::split, bp::args("self", "x"),
          (split_doc +
           " This returns a list of mutable references to each component.")
              .c_str())
      .def(
          "split_vector",
          +[](CartesianProduct const &m, const ConstVectorRef &x) {
            return copy_vec_constref(m.split_vector(x));
          },
          bp::args("self", "v"), split_vec_doc.c_str())
      .def<MutSplitSig>(
          "split_vector", &CartesianProduct::split_vector,
          bp::args("self", "v"),
          (split_vec_doc +
           " This returns a list of mutable references to each component.")
              .c_str())
      .def("merge", &CartesianProduct::merge, bp::args("self", "xs"),
           "Define a point on the manifold by merging points from each "
           "component.")
      .def("merge_vector", &CartesianProduct::merge_vector,
           bp::args("self", "vs"),
           "Define a tangent vector on the manifold by merging vectors from "
           "each component.")
      .def(
          "__mul__", +[](const shared_ptr<CartesianProduct> &a,
                         const ManifoldPtr &b) { return a * b; });
}

} // namespace proxnlp::python
