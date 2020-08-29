#ifndef PROBLEM_H
#define PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>

#include "edge.h"
#include "vertex.h"

typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem {
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Problem();

    ~Problem();

    bool AddVertex(std::shared_ptr<Vertex> vertex);
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);
    bool AddEdge(std::shared_ptr<Edge> edge);
    bool RemoveEdge(std::shared_ptr<Edge> edge);

    bool Solve(int iterations = 10);

private:

    // 获取某个顶点连接到的边
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    // 设置各顶点的ordering_index
    void SetOrdering();
    // 构造大H矩阵
    void MakeHessian();
    // Levenberg
    // 计算LM算法的初始Lambda
    void ComputeLambdaInitLM();
    // 解线性方程
    void SolveLinearSystem();
    // 更新状态变量
    void UpdateStates();
    // 有时候 update 后残差会变大，需要退回去，重来
    void RollbackStates(); 
    // LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
    bool IsGoodStepInLM();

    double currentLambda_;
    double currentChi_;
    double stopThresholdLM_;    // LM 迭代退出阈值条件
    double ni_;                 //控制 Lambda 缩放大小
    
    // 整个信息矩阵
    Eigen::MatrixXd Hessian_;
    Eigen::VectorXd b_;
    Eigen::VectorXd delta_x_;

    // 先验部分信息
    Eigen::MatrixXd H_prior_;
    Eigen::VectorXd b_prior_;
    Eigen::VectorXd b_prior_backup_;
    Eigen::VectorXd err_prior_;
    Eigen::VectorXd err_prior_backup_;
    Eigen::MatrixXd Jt_prior_inv_;

    // Ordering related
    unsigned long ordering_poses_ = 0;
    unsigned long ordering_landmarks_ = 0;
    unsigned long ordering_generic_ = 0;

    // all edges
    HashEdge edges_;
    // 由vertex id查询edge
    HashVertexIdToEdge vertexToEdge_;

    // all vertices
    HashVertex verticies_;
    // verticies need to marg. <Ordering_id_, Vertex>
    HashVertex verticies_marg_;

};

#endif
