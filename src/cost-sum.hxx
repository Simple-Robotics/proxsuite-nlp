#pragma once

#include "proxnlp/cost-sum.hpp"


namespace proxnlp
{

  template<typename Scalar>
  CostSum<Scalar> operator+(const CostFunctionBaseTpl<Scalar>& left, const CostFunctionBaseTpl<Scalar>& right)
  {
    assert((left.nx() == right.nx()) && (left.ndx() == right.ndx()) &&
           "Left and right should have the same input spaces.");
    CostSum<Scalar> out(left.nx(), left.ndx());
    out += left;
    out += right;
    return out;
  }

  // left is rvalue reference, so we modify it, return a move of the left after adding right
  template<typename Scalar>
  CostSum<Scalar>&& operator+(CostSum<Scalar>&& left, const CostFunctionBaseTpl<Scalar>& right)
  {
    left += right;
    return std::move(left);
  }

  // create a CostSum object with the desired weight
  template<typename Scalar>
  CostSum<Scalar> operator*(const Scalar& left, const CostFunctionBaseTpl<Scalar>& right)
  {
    CostSum<Scalar> out(right.nx(), right.ndx());
    out.addComponent(right, left);
    return out;
  }

  template<typename Scalar>
  CostSum<Scalar>&& operator*(Scalar&& left, CostSum<Scalar>&& right)
  {
    right *= left;
    return std::move(right);
  }

  template<typename Scalar>
  inline
  CostSum<Scalar>&& operator*(CostSum<Scalar>&& left, Scalar&& right)
  {
    return right * left;
  }
} // namespace proxnlp
