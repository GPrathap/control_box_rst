#include <iostream>
#include <vector>
#include <cmath>
#include <matplotlibcpp.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/vertices/vertex_se2.h>
#include <g2o/edge_se2.h>
#include <g2o/edge_pointxy.h>

namespace plt = matplotlibcpp;

struct State {
    double x;
    double y;
    double theta;
    double v;
};

struct ControlInput {
    double a;      // acceleration
    double delta;  // steering angle
};

// Generate a circular reference trajectory
std::vector<State> generateCircularTrajectory(double radius, double centerX, double centerY, int numPoints) {
    std::vector<State> trajectory;
    for (int i = 0; i < numPoints; ++i) {
        double theta = 2.0 * M_PI * i / numPoints;
        State refState;
        refState.x = centerX + radius * cos(theta);
        refState.y = centerY + radius * sin(theta);
        refState.theta = theta + M_PI / 2.0; // Orientation tangent to the circle
        refState.v = 1.0;  // Constant speed
        trajectory.push_back(refState);
    }
    return trajectory;
}

class MPCOptimization {
public:
    MPCOptimization() {
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>> BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

        std::unique_ptr<BlockSolverType> solver_ptr(new BlockSolverType(std::unique_ptr<LinearSolverType>(new LinearSolverType())));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
        optimizer.setAlgorithm(solver);
    }

    void addStateVertex(int id, const State& state) {
        g2o::VertexSE2* v = new g2o::VertexSE2();
        v->setId(id);
        v->setEstimate(g2o::SE2(state.x, state.y, state.theta));
        optimizer.addVertex(v);
        optimizedStates.push_back(state);
    }

    void addControlEdge(int from, int to, const ControlInput& u, double dt, double L) {
        g2o::EdgeSE2* e = new g2o::EdgeSE2();
        e->vertices()[0] = optimizer.vertex(from);
        e->vertices()[1] = optimizer.vertex(to);

        State predictedState = predictState(u, dt, L);
        g2o::SE2 measurement(predictedState.x, predictedState.y, predictedState.theta);
        e->setMeasurement(measurement);
        e->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(e);
        optimizedStates[to] = predictedState;
    }

    void addTrackingEdge(int id, const State& refState) {
        g2o::EdgeSE2PointXY* e = new g2o::EdgeSE2PointXY();
        e->vertices()[0] = optimizer.vertex(id);
        Eigen::Vector2d reference(refState.x, refState.y);
        e->setMeasurement(reference);
        e->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(e);
    }

    void optimize() {
        optimizer.initializeOptimization();
        optimizer.optimize(10);
    }

    std::vector<State> getOptimizedStates() const {
        return optimizedStates;
    }

private:
    g2o::SparseOptimizer optimizer;
    std::vector<State> optimizedStates;

    State predictState(const ControlInput& u, double dt, double L) {
        State newState;
        double x = optimizedStates.back().x;
        double y = optimizedStates.back().y;
        double theta = optimizedStates.back().theta;
        double v = optimizedStates.back().v;

        newState.x = x + v * cos(theta) * dt;
        newState.y = y + v * sin(theta) * dt;
        newState.theta = theta + (v / L) * tan(u.delta) * dt;
        newState.v = v + u.a * dt;

        return newState;
    }
};

int main() {
    MPCOptimization mpc;

    // Generate a reference circular trajectory
    double radius = 5.0;
    double centerX = 0.0;
    double centerY = 0.0;
    int numPoints = 100;
    std::vector<State> refTrajectory = generateCircularTrajectory(radius, centerX, centerY, numPoints);

    // Initial state
    State initialState = {5.0, 0.0, M_PI / 2.0, 1.0};
    mpc.addStateVertex(0, initialState);

    // Horizon parameters
    int horizonLength = 10;
    double dt = 0.1;
    double L = 2.0;

    // Multiple Shooting: Add state vertices and control edges
    for (int i = 1; i <= horizonLength; ++i) {
        State predictedState = initialState; // Placeholder, will be updated
        mpc.addStateVertex(i, predictedState);

        // Control input (example values, should be optimized)
        ControlInput u = {0.1, 0.05};
        mpc.addControlEdge(i - 1, i, u, dt, L);

        // Tracking the reference trajectory
        int refIdx = (i * numPoints) / horizonLength;
        mpc.addTrackingEdge(i, refTrajectory[refIdx]);
    }

    // Optimize the trajectory
    mpc.optimize();

    // Retrieve optimized states and plot
    std::vector<State> optimizedStates = mpc.getOptimizedStates();

    // Prepare data for plotting
    std::vector<double> refX, refY, optX, optY;
    for (const auto& state : refTrajectory) {
        refX.push_back(state.x);
        refY.push_back(state.y);
    }
    for (const auto& state : optimizedStates) {
        optX.push_back(state.x);
        optY.push_back(state.y);
    }

    // Plot the reference trajectory and optimized trajectory
    plt::figure();
    plt::plot(refX, refY, "r--", {{"label", "Reference Trajectory"}});
    plt::plot(optX, optY, "b-", {{"label", "Optimized Trajectory"}});
    plt::legend();
    plt::xlabel("X Position");
    plt::ylabel("Y Position");
    plt::title("MPC Trajectory Tracking");

    plt::show();

    return 0;
}
