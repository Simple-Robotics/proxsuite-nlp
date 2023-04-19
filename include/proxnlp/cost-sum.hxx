/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/cost-sum.hpp"

namespace proxnlp {

template <typename Scalar>
auto operator+(const shared_ptr<CostFunctionBaseTpl<Scalar>> &left,
               const shared_ptr<CostFunctionBaseTpl<Scalar>> &right) {
  assert((left->nx() == right->nx()) && (left->ndx() == right->ndx()) &&
         "Left and right should have the same input spaces.");
  auto out = std::make_shared<CostSumTpl<Scalar>>(left->nx(), left->ndx());
  *out += left;
  *out += right;
  return out;
}

// left is rvalue reference, so we modify it, return a move of the left after
// adding right
template <typename Scalar>
auto &&operator+(shared_ptr<CostSumTpl<Scalar>> &&left,
                 const shared_ptr<CostFunctionBaseTpl<Scalar>> &right) {
  *left += right;
  return std::move(left);
}

// create a CostSum object with the desired weight
template <typename Scalar>
auto operator*(Scalar left,
               const shared_ptr<CostFunctionBaseTpl<Scalar>> &right) {
  auto out = std::make_shared<CostSumTpl<Scalar>>(right->nx(), right->ndx());
  out->addComponent(right, left);
  return out;
}
} // namespace proxnlp
