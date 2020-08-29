#include <iostream>
#include <random>

#include "problem.h"
#include <glog/logging.h>

using namespace std;

class CurveFittingVertex: public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingVertex(): Vertex(3) {}
    virtual std::string TypeInfo() const { return "abc"; }
};

class CurveFittingEdge: public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x, double y): Edge(1, 1, std::vector<std::string>{"abc"}) {
        x_ = x;
        y_ = y;
    }
    virtual void ComputeResidual() override {
        Eigen::Vector3d abc = verticies_[0]->Parameters();
        residual_(0) = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2)) - y_;
    }
    virtual void ComputeJacobians() override {
        Eigen::Vector3d abc = verticies_[0]->Parameters();
        double exp_y = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));
        Eigen::Matrix<double, 1, 3> jaco_abc;
        jaco_abc << x_ * x_ * exp_y, x_ * exp_y , 1 * exp_y;
        jacobians_[0] = jaco_abc;
    }
    virtual std::string TypeInfo() const override { return "CurveFittingEdge"; }
public:
    double x_, y_;
};

int main(int argc, char** argv) {

    FLAGS_alsologtostderr = true;
	google::InitGoogleLogging(argv[0]);

    double a = 1.0, b = 2.0, c = 1.0;   // 真实参数值
    int N = 100;                        // 数据点
    double w_sigma= 1.;                 // 噪声Sigma值

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0., w_sigma);

    Problem problem;
    shared_ptr<CurveFittingVertex> vertex(new CurveFittingVertex());

    vertex->SetParameters(Eigen::Vector3d(0., 0., 0.));
    problem.AddVertex(vertex);

    for (int i = 0; i < N; ++i) {

        double x = i / 100.;
        double n = noise(generator);
        double y = std::exp(a * x * x + b * x + c) + n;

        shared_ptr<CurveFittingEdge> edge(new CurveFittingEdge(x, y));
        std::vector<std::shared_ptr<Vertex>> edge_vertex;
        edge_vertex.push_back(vertex);
        edge->SetVertex(edge_vertex);
        problem.AddEdge(edge);
    }

    LOG(INFO) << "Test CurveFitting start...\n";
    problem.Solve(30);
    LOG(INFO) << "-------The optimized parameters:\n";
    LOG(INFO) << vertex->Parameters().transpose() << std::endl;
    LOG(INFO) << "-------ground truth:\n";
    LOG(INFO) << "1.0,  2.0,  1.0\n";

    return 0;
}
