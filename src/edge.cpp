#include "vertex.h"
#include "edge.h"
#include <glog/logging.h>

unsigned long global_edge_id = 0;

Edge::Edge(int residual_dimension, int num_verticies,
           const std::vector<std::string> &verticies_types) {
    residual_.resize(residual_dimension, 1);
    if (!verticies_types.empty())
        verticies_types_ = verticies_types;
    jacobians_.resize(num_verticies);
    id_ = global_edge_id++;

    Eigen::MatrixXd information(residual_dimension, residual_dimension);
    information.setIdentity();
    information_ = information;

    lossfunction_ = nullptr;
}

Edge::~Edge() {}

double Edge::Chi2() const{
    return residual_.transpose() * information_ * residual_;
}

double Edge::RobustChi2() const{
    double e2 = this->Chi2();
    if(lossfunction_) {
        Eigen::Vector3d rho;
        lossfunction_->Compute(e2,rho);
        e2 = rho[0];
    }
    return e2;
}

void Edge::RobustInfo(double &drho, Eigen::MatrixXd &info) const{
    if (lossfunction_) {
        double e2 = this->Chi2();
        Eigen::Vector3d rho;
        lossfunction_->Compute(e2,rho);
        Eigen::VectorXd weight_err = information_ * residual_;

        Eigen::MatrixXd robust_info(information_.rows(), information_.cols());
        robust_info.setIdentity();
        robust_info *= rho[1] * information_;
        if(rho[1] + 2 * rho[2] * e2 > 0.) {
            robust_info += 2 * rho[2] * weight_err * weight_err.transpose();
        }
        info = robust_info;
        drho = rho[1];
    }
    else {
        drho = 1.0;
        info = information_;
    }
}
