#ifndef EDGE_H
#define EDGE_H

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "loss.h"

class Vertex;

/**
 * 边负责计算残差，残差是 预测-观测，维度在构造函数中定义
 * 代价函数是 残差*信息*残差，是一个数值，由后端求和后最小化
 */
class Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * 构造函数，会自动化配雅可比的空间
     * @param residual_dimension 残差维度
     * @param num_verticies 顶点数量
     * @param verticies_types 顶点类型名称，可以不给，不给的话check中不会检查
     */
    explicit Edge(int residual_dimension, int num_verticies,
        const std::vector<std::string> &verticies_types = std::vector<std::string>());
    virtual ~Edge();

    // 返回id
    unsigned long Id() const { return id_; }

    /**
     * 设置一个顶点
     * @param vertex 对应的vertex对象
     */
    bool AddVertex(std::shared_ptr<Vertex> vertex) {
        verticies_.emplace_back(vertex);
        return true;
    }

    /**
     * 设置一些顶点
     * @param vertices 顶点，按引用顺序排列
     * @return
     */
    bool SetVertex(const std::vector<std::shared_ptr<Vertex>> &vertices) {
        verticies_ = vertices;
        return true;
    }

    // 返回第i个顶点
    std::shared_ptr<Vertex> GetVertex(int i) {
        return verticies_[i];
    }

    // 返回所有顶点
    std::vector<std::shared_ptr<Vertex>> Verticies() const {
        return verticies_;
    }

    // 返回关联顶点个数
    size_t NumVertices() const { return verticies_.size(); }

    // 返回边的类型信息，在子类中实现
    virtual std::string TypeInfo() const = 0;

    // 计算残差，由子类实现
    virtual void ComputeResidual() = 0;

    // 计算雅可比，由子类实现
    virtual void ComputeJacobians() = 0;

    // 计算平方误差，会乘以信息矩阵
    double Chi2() const;
    double RobustChi2() const;

    // 返回残差
    Eigen::VectorXd Residual() const { return residual_; }

    // 返回雅可比
    std::vector<Eigen::MatrixXd> Jacobians() const { return jacobians_; }

    // 设置信息矩阵
    void SetInformation(const Eigen::MatrixXd &information) {
        information_ = information;
        sqrt_information_ = Eigen::LLT<Eigen::MatrixXd>(information_).matrixL().transpose();
    }

    // 返回信息矩阵
    Eigen::MatrixXd Information() const {
        return information_;
    }

    Eigen::MatrixXd SqrtInformation() const {
        return sqrt_information_;
    }

    void SetLossFunction(LossFunction* ptr){ lossfunction_ = ptr; }
    LossFunction* GetLossFunction(){ return lossfunction_;}
    void RobustInfo(double& drho, Eigen::MatrixXd& info) const;

    // 设置观测信息
    void SetObservation(const Eigen::VectorXd& observation) {
        observation_ = observation;
    }

    // 返回观测信息
    Eigen::VectorXd Observation() const { return observation_; }

    int OrderingId() const { return ordering_id_; }
    void SetOrderingId(int id) { ordering_id_ = id; };

protected:
    unsigned long id_;  // edge id
    int ordering_id_;   // edge id in problem
    std::vector<std::string> verticies_types_;  // 各顶点类型信息，用于debug
    std::vector<std::shared_ptr<Vertex>> verticies_; // 该边对应的顶点
    Eigen::VectorXd residual_;                 // 残差
    std::vector<Eigen::MatrixXd> jacobians_;  // 雅可比，每个雅可比维度是 residual x vertex[i]
    Eigen::MatrixXd information_;             // 信息矩阵
    Eigen::MatrixXd sqrt_information_;
    Eigen::VectorXd observation_;              // 观测信息

    LossFunction *lossfunction_;
};

#endif
