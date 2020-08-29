#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

/**
 * Base class
 * 
 * compute the scaling factor for a error:
 * The error is e^T Omega e
 * The output rho is
 * rho[0]: The actual scaled error value
 * rho[1]: First derivative of the scaling function
 * rho[2]: Second derivative of the scaling function
 *
*/
class LossFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual ~LossFunction() {}
    virtual void Compute(double err2, Eigen::Vector3d& rho) const = 0;
};

/**
 * Trival loss
 * 
 * TrivalLoss(e) = e^2
 */
class TrivalLoss : public LossFunction {
public:
    virtual void Compute(double err2, Eigen::Vector3d& rho) const override {
        rho[0] = err2;
        rho[1] = 1;
        rho[2] = 0;
    }
};

/**
 * Huber loss
 *
 * Huber(e) = e^2                      if e <= delta
 * Huber(e) = delta*(2*e - delta)      if e > delta
 * 
 */
class HuberLoss : public LossFunction {
public:
    explicit HuberLoss(double delta) : delta_(delta) {}
    virtual void Compute(double err2, Eigen::Vector3d& rho) const override;
private:
    double delta_;
};

/**
 * Cauchy loss
 * 
 * c = delta
 * Cauchy(e) = c^2 * log( 1 + e^2/c^2 )
 * 
 */
class CauchyLoss : public LossFunction {
public:
    explicit CauchyLoss(double delta) : delta_(delta) {}
    virtual void Compute(double err2, Eigen::Vector3d& rho) const override;

private:
    double delta_;
};

/**
 * Tukey loss
 * 
 * Tukey(e) = delta^2 / 6 * (1 - (1 - e^2 / delta^2)^3)     if e <= delta
 * Tukey(e) = delta^2 / 6                                   if e > delta
 * 
 */
class TukeyLoss : public LossFunction {
public:
    explicit TukeyLoss(double delta) : delta_(delta) {}
    virtual void Compute(double err2, Eigen::Vector3d& rho) const override;

private:
    double delta_;
};

#endif
