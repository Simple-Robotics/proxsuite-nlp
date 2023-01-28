/// @file
/// @brief Test that CNPY does load test files properly.
#include "cnpy.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(numpy_load_mat) {

  std::string fname = "npy_payload.npy";

  BOOST_TEST_MESSAGE("Compile-time matrix:");
  auto load_mat = cnpy::npy_load_mat<double>(fname);
  std::cout << load_mat << std::endl;
  BOOST_TEST_CHECK(load_mat.rows() == 3);
  BOOST_TEST_CHECK(load_mat.cols() == 4);
}

BOOST_AUTO_TEST_CASE(numpy_load_vec) {

  std::string fname2 = "npy_payload2.npy";

  BOOST_TEST_MESSAGE("Compile-time vector:");
  auto load_vec = cnpy::npy_load_vec<double>(fname2);
  std::cout << load_vec << std::endl;

  BOOST_TEST_CHECK(load_vec.rows() == 6);
}
