#ifndef VERTEX_H
#define VERTEX_H

#include <Eigen/Dense>

extern unsigned long global_vertex_id;

class Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * 构造函数
     * @param num_dimension 顶点自身维度
     * @param local_dimension 本地参数化维度，为-1时认为与本身维度一样
     */
    explicit Vertex(int num_dimension, int local_dimension = -1);
    virtual ~Vertex();

    int Dimension() const;
    int LocalDimension() const;
    unsigned long Id() const { return id_; }

    Eigen::VectorXd Parameters() const { return parameters_; }
    Eigen::VectorXd& Parameters() { return parameters_; }
    void SetParameters(const Eigen::VectorXd& params) { parameters_ = params; }

    void BackUpParameters() { parameters_backup_ = parameters_; }
    void RollBackParameters() { parameters_ = parameters_backup_; }

    virtual void Plus(const Eigen::VectorXd& delta);
    virtual std::string TypeInfo() const = 0;

    int OrderingId() const { return ordering_id_; }
    void SetOrderingId(unsigned long id) { ordering_id_ = id; };

    void SetFixed(bool fixed = true) {
        fixed_ = fixed;
    }
    bool IsFixed() const { return fixed_; }

protected:
    Eigen::VectorXd parameters_;   // 实际存储的变量值
    Eigen::VectorXd parameters_backup_; // 每次迭代优化中对参数进行备份，用于回滚
    int local_dimension_;   // 局部参数化维度
    unsigned long id_;  // 顶点的id，自动生成

    // ordering id是在problem中排序后的id，用于寻找雅可比对应块
    // ordering id带有维度信息，例如ordering_id=6则对应Hessian中的第6列
    // 从零开始
    unsigned long ordering_id_ = 0;
    bool fixed_ = false;    // 是否固定
};

#endif
