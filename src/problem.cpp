#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "problem.h"

#include <glog/logging.h>

Problem::Problem() {
    verticies_marg_.clear();
}

Problem::~Problem() {
    global_vertex_id = 0;
}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(std::make_pair(vertex->Id(), vertex));
    }
    return true;
}

bool Problem::AddEdge(std::shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(std::make_pair(edge->Id(), edge));
    } else {
        LOG(WARNING) << "Edge " << edge->Id() << " has been added before!\n";
        return false;
    }

    for (auto &vertex: edge->Verticies()) {
        vertexToEdge_.insert(std::make_pair(vertex->Id(), edge));
    }
    return true;
}

std::vector<std::shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
    std::vector<std::shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {
        // 这个edge还存在，而不是已经被remove了
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;
        edges.emplace_back(iter->second);
    }
    return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!\n";
        return false;
    }
    std::vector<std::shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        RemoveEdge(remove_edges[i]);
    }
    vertex->SetOrderingId(-1);      // NOTE used to debug
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());
    return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!\n";
        return false;
    }
    edges_.erase(edge->Id());
    return true;
}


bool Problem::Solve(int iterations) {

    if (edges_.size() == 0 || verticies_.size() == 0) {
        LOG(ERROR) << "\nCannot solve problem without edges or verticies\n";
        return false;
    }

    // 统计优化变量的维数，为构建 H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建 H 矩阵
    MakeHessian();
    // LM 初始化
    ComputeLambdaInitLM();
    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    double last_chi_ = 1e20;
    while (!stop && (iter < iterations)) {
        LOG(INFO) << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一步
        {
            // setLambda
            // 第四步，解线性方程
            SolveLinearSystem();
            // 更新状态量
            UpdateStates();
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
            oneStepSuccess = IsGoodStepInLM();
            // 后续处理，
            if (oneStepSuccess) {
                LOG(INFO) << "get one step success\n";
                // 在新线性化点 构建 hessian
                MakeHessian();
                false_cnt = 0;
            } else {
                false_cnt ++;
                RollbackStates();   // 误差没下降，回滚
            }
        }
        iter++;

        if (last_chi_ - currentChi_ < 1e-5) {
            LOG(INFO) << "last_chi_ - currentChi_ < 1e-5";
            stop = true;
        }
        last_chi_ = currentChi_;
    }
    return true;
}

void Problem::SetOrdering() {

    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;
    for (auto vertex: verticies_)
        ordering_generic_ += vertex.second->LocalDimension();

}

void Problem::MakeHessian() {

    // 直接构造大的 H 矩阵
    unsigned long size = ordering_generic_;
    Eigen::MatrixXd H(Eigen::MatrixXd::Zero(size, size));
    Eigen::VectorXd b(Eigen::VectorXd::Zero(size));

    for (auto &edge: edges_) {

        edge.second->ComputeResidual();
        edge.second->ComputeJacobians();

        auto jacobians = edge.second->Jacobians();
        auto verticies = edge.second->Verticies();
        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            unsigned long index_i = v_i->OrderingId();
            unsigned long dim_i = v_i->LocalDimension();

            // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
            double drho;
            Eigen::MatrixXd robustInfo(edge.second->Information().rows(),edge.second->Information().cols());
            edge.second->RobustInfo(drho,robustInfo);

            Eigen::MatrixXd JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->IsFixed()) continue;

                auto jacobian_j = jacobians[j];
                unsigned long index_j = v_j->OrderingId();
                unsigned long dim_j = v_j->LocalDimension();

                assert(v_j->OrderingId() != -1);
                Eigen::MatrixXd hessian = JtW * jacobian_j;

                // 所有的信息矩阵叠加起来
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

                }
            }
            b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge.second->Information() * edge.second->Residual();
        }

    }
    Hessian_ = H;
    b_ = b;

    if(H_prior_.rows() > 0) {
        Eigen::MatrixXd H_prior_tmp = H_prior_;
        Eigen::VectorXd b_prior_tmp = b_prior_;

        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }

    delta_x_ = Eigen::VectorXd::Zero(size);  // initial delta_x = 0_n;

}

void Problem::ComputeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;

    for (auto edge: edges_) {
        currentChi_ += edge.second->RobustChi2();
    }
    if (err_prior_.rows() > 0)
        currentChi_ += err_prior_.squaredNorm();
    currentChi_ *= 0.5;

    stopThresholdLM_ = 1e-10 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

    double maxDiagonal = 0;
    unsigned long size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (unsigned long i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
    }

    maxDiagonal = std::min(5e10, maxDiagonal);
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal;
    LOG(INFO) << "currentLamba_: "<<maxDiagonal<<" "<<currentLambda_<<std::endl;
}

void Problem::SolveLinearSystem() {
    Eigen::MatrixXd H = Hessian_;
    for (size_t i = 0; i < Hessian_.cols(); ++i)
        H(i, i) += currentLambda_;
    delta_x_ = H.ldlt().solve(b_);
}

void Problem::UpdateStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->BackUpParameters();    // 保存上次的估计值

        unsigned long idx = vertex.second->OrderingId();
        unsigned long dim = vertex.second->LocalDimension();
        Eigen::VectorXd delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }

    // update prior
    if (err_prior_.rows() > 0) {
        // BACK UP b_prior_
        b_prior_backup_ = b_prior_;
        err_prior_backup_ = err_prior_;

        /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
        /// \delta x = Computes the linearized deviation from the references (linearization points)
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);       // update the error_prior
        err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);
        
        LOG(INFO) << "                : "<< b_prior_.norm()<<" " <<err_prior_.norm()<< std::endl;
        LOG(INFO) << "     delta_x_ ex: "<< delta_x_.head(6).norm() << std::endl;
    }

}

void Problem::RollbackStates() {

    for (auto vertex: verticies_) {
        vertex.second->RollBackParameters();
    }

    if (err_prior_.rows() > 0) {
        b_prior_ = b_prior_backup_;
        err_prior_ = err_prior_backup_;
    }
}


bool Problem::IsGoodStepInLM() {
    double scale = 0;
    scale = 0.5* delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale += 1e-6;    // make sure it's non-zero :)

    // recompute residuals after update state
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->RobustChi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.squaredNorm();
    tempChi *= 0.5;          // 1/2 * err^2

    double rho = (currentChi_ - tempChi) / scale;
    if (rho > 0 && std::isfinite(tempChi))   // last step was good, 误差在下降
    {
        double alpha = 1. - pow((2 * rho - 1), 3);
        alpha = std::min(alpha, 2. / 3.);
        double scaleFactor = (std::max)(1. / 3., alpha);
        currentLambda_ *= scaleFactor;
        ni_ = 2;
        currentChi_ = tempChi;
        return true;
    } else {
        currentLambda_ *= ni_;
        ni_ *= 2;
        return false;
    }
}
