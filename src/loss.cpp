#include <cmath>
#include "loss.h"

void HuberLoss::Compute(double e2, Eigen::Vector3d& rho) const {
    double dsqr = delta_ * delta_;
    if (e2 <= dsqr) {
        rho[0] = e2;
        rho[1] = 1.;
        rho[2] = 0.;
    }
    else {
        double e = std::sqrt(e2);
        rho[0] = 2 * e * delta_ - dsqr;
        rho[1] = delta_ / e;
        rho[2] = -0.5 * rho[1] / e2;
    }
}

void CauchyLoss::Compute(double e2, Eigen::Vector3d& rho) const {
    double dsqr = delta_ * delta_;
    double dsqrReci = 1. / dsqr;
    double aux = dsqrReci * e2 + 1.0;
    rho[0] = dsqr * std::log(aux);
    rho[1] = 1. / aux;
    rho[2] = -dsqrReci * std::pow(rho[1], 2);
}

void TukeyLoss::Compute(double e2, Eigen::Vector3d& rho) const
{
    const double e = std::sqrt(e2);
    const double delta2 = delta_ * delta_;
    if (e <= delta_) {
        const double aux = e2 / delta2;
        rho[0] = delta2 * (1. - std::pow((1. - aux), 3)) / 3.;
        rho[1] = std::pow((1. - aux), 2);
        rho[2] = -2. * (1. - aux) / delta2;
    }
    else {
        rho[0] = delta2 / 3.;
        rho[1] = 0;
        rho[2] = 0;
    }
}
