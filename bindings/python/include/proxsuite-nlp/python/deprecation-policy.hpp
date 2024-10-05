//
// Copyright (C) 2024 LAAS-CNRS, INRIA
//
#pragma once

#include "proxsuite-nlp/deprecated.hpp"
#include <eigenpy/deprecation-policy.hpp>

PROXSUITE_NLP_DEPRECATED_HEADER("This header has been deprecated. Use "
                                "<eigenpy/deprecation-policy.hpp> instead.")

namespace proxsuite {
namespace nlp {
using eigenpy::deprecated_function;
using eigenpy::deprecated_member;
using eigenpy::deprecation_warning_policy;
using eigenpy::DeprecationType;
} // namespace nlp
} // namespace proxsuite
