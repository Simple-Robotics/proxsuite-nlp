#include "proxsuite-nlp/results.hpp"
#include "proxsuite-nlp/workspace.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ResultsTpl<context::Scalar>;

}
} // namespace proxsuite
