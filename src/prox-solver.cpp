#include "proxsuite-nlp/prox-solver.hpp"
#include "proxsuite-nlp/modelling/spaces/vector-space.hpp"

namespace proxsuite {
namespace nlp {

template class PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    ProxNLPSolverTpl<context::Scalar>;

}
} // namespace proxsuite
