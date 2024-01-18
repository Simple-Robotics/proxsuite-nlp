/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "proxsuite-nlp/pdal.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ALMeritFunctionTpl<context::Scalar>;

}
} // namespace proxsuite
