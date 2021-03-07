
#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <tuple>
#include <vector>

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#define EIGEN_DENSEBASE_PLUGIN "eigen_serialization.h"
#include <Eigen/Core>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

// shortening used eigen class' names
using matrix = Eigen::MatrixXf;
using vector = Eigen::VectorXf;

// input and corresponding correct output vector
struct instance {
    vector input, output;
};

using data_set = std::vector<instance>;


class neural_net {
public:

    explicit neural_net(const std::vector<unsigned> &topology);
    ~neural_net() = default;

    [[nodiscard]] vector feed_forward(const vector &input) const;

    // splits training_set into batches of batch_size and for each calls
    // propagate_batch
    void train(const data_set &training_set, unsigned epochs, unsigned batch_size,
               float learning_rate);
    [[nodiscard]] float mean_squared_error(const data_set &test_set) const;

    [[nodiscard]] std::vector<matrix> get_weights() const { return m_weights; }
    [[nodiscard]] std::vector<vector> get_biases() const { return m_biases; }
    [[nodiscard]] std::vector<unsigned> get_topology() const { return m_topology; }

    template <class Archive>
    friend Archive &operator<<(Archive &a, const neural_net &n) {
        a << n.m_biases << n.m_weights << n.m_topology;
        return a;
    }

    template <class Archive>
    friend Archive &operator>>(Archive &a, neural_net &n) {
        a >> n.m_biases >> n.m_weights >> n.m_topology;
        return a;
    }

    static neural_net deserialize(const std::string &file_path);

    static void serialize(const neural_net &nn, const std::string &file_path);

private:
    neural_net() = default;

    // starting from start_index in training_set, for each InOutPair calls
    // propagate_back and averages delta values
    void propagate_batch(const data_set &training_set, unsigned start_index,
                         unsigned batch_size, float learning_rate);

    // returns changes for each weight matrix and change for each bias matrix
    [[nodiscard]] std::tuple<std::vector<matrix>, std::vector<vector>>
    get_delta_weights_and_biases(const vector &input,
                                 const vector &desired_output,
                                 float learning_rate) const;

    // activation functions and their derivatives
    static float sigmoid(float x);
    static float d_sigmoid(float x);

    std::vector<matrix> m_weights;
    std::vector<vector> m_biases;
    std::vector<unsigned> m_topology;
};



#endif
