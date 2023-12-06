#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <proxnlp/modelling/spaces/multibody.hpp>
#include <proxnlp/modelling/costs/quadratic-residual.hpp>
#include <proxnlp/modelling/costs/squared-distance.hpp>
#include <proxnlp/cost-sum.hpp>
#include <proxnlp/solver-base.hpp>
#include <proxnlp/modelling/constraints.hpp>

#include <boost/filesystem/path.hpp>

using Scalar = double;
PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

using namespace proxnlp;
namespace pin = pinocchio;

using Model = pin::ModelTpl<Scalar>;
using Data = pin::DataTpl<Scalar>;
using Space = MultibodyConfiguration<Scalar>;
using Cost = CostFunctionBaseTpl<Scalar>;
const boost::filesystem::path URDF_PATH(EXAMPLE_ROBOT_DATA_MODEL_DIR);

Model loadModel() {
  auto path = URDF_PATH;
  path /= "ur_description/urdf/ur5_robot.urdf";
  Model out;
  pin::urdf::buildModel(path.string(), out);
  return out;
}

struct FramePosition : C2FunctionTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = C2FunctionTpl<Scalar>;

  shared_ptr<Space> space_;
  mutable Data data_;
  pin::FrameIndex fid_;
  Vector3s ref_;
  mutable Matrix6Xs fJf_;

  FramePosition(const shared_ptr<Space> &space, pin::FrameIndex fid,
                const Vector3s &ref)
      : Base(*space, 3), space_(space), data_(space->getModel()), fid_(fid),
        ref_(ref), fJf_(6, getModel().nv) {
    fJf_.setZero();
  }

  Model const &getModel() const { return space_->getModel(); }

  VectorXs operator()(const ConstVectorRef &q) const override {
    pin::forwardKinematics(getModel(), data_, q);
    pin::updateFramePlacement(getModel(), data_, fid_);
    return data_.oMf[fid_].translation() - ref_;
  }

  void computeJacobian(const ConstVectorRef & /*q*/,
                       MatrixRef Jout) const override {
    pin::computeJointJacobians(getModel(), data_);
    pin::getFrameJacobian(getModel(), data_, fid_, pin::LOCAL, fJf_);
    Jout.leftCols(getModel().nv) = data_.oMf[fid_].rotation() * fJf_.topRows(3);
  }
};

int main() {

  const std::string ee_link_name = "tool0";
  Model model = loadModel();
  auto space = std::make_shared<Space>(model);
  pin::FrameIndex fid = model.getFrameId(ee_link_name);

  Vector3s ref(1.0, 0., 0.2);
  auto fn = allocate_shared_eigen_aligned<FramePosition>(space, fid, ref);

  auto q0 = pin::neutral(model);
  MatrixXs w1(3, 3);
  w1.setIdentity();
  w1 *= 4.0;
  MatrixXs w2(model.nv, model.nv);
  w2.setIdentity();
  w2 *= 1e-2;

  auto cost1 =
      allocate_shared_eigen_aligned<QuadraticResidualCostTpl<Scalar>>(fn, w1);
  auto cost2 = allocate_shared_eigen_aligned<QuadraticDistanceCostTpl<Scalar>>(
      space, q0, w2);
  auto cost = std::make_shared<CostSumTpl<Scalar>>(space->nx(), space->ndx());
  cost->addComponent(cost1);
  cost->addComponent(cost2);

  auto problem = std::make_shared<ProblemTpl<Scalar>>(space, cost);

  constexpr bool has_joint_lims = true;
  if (has_joint_lims) {
    auto box_cstr = std::make_shared<BoxConstraintTpl<Scalar>>(
        model.lowerPositionLimit, model.upperPositionLimit);
    ConstraintObjectTpl<Scalar> cstrobj(
        std::make_shared<ManifoldDifferenceToPoint<Scalar>>(space, q0),
        box_cstr);
    problem->addConstraint(cstrobj);
  }

  SolverTpl<Scalar> solver(problem, 1e-4, 0.01, 0.0, proxnlp::VERBOSE);
  solver.setup();
  solver.solve(q0);

  ResultsTpl<Scalar> const &results = solver.getResults();
  WorkspaceTpl<Scalar> const &ws = solver.getWorkspace();
  std::cout << results << std::endl;

  fmt::print("Optimal cost: {}\n", ws.objective_value);
  fmt::print("Optimal cost grad: {}\n", ws.objective_gradient.transpose());
  auto opt_frame_pos = (*fn)(results.x_opt);
  fmt::print("Optimal frame pos: {}\n", opt_frame_pos.transpose());

  return 0;
}
