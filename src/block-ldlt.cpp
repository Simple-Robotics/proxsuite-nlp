#include "proxsuite-nlp/linalg/block-ldlt.hpp"

namespace proxsuite {
namespace nlp {
namespace linalg {

template struct PROXSUITE_NLP_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    BlockLDLT<context::Scalar>;

} // namespace linalg
} // namespace nlp
} // namespace proxsuite
