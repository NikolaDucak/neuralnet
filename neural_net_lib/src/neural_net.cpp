#include "../inc/neural_net.h"
#include <fstream>

neural_net::neural_net(const std::vector<unsigned> &topology) :
        m_topology(topology) {
    // skip l=0 layer, input layer has no m_biases
    // and m_weights connecting that layer to next are stored at index of next layer
    // l=1
    for (unsigned l = 1; l < topology.size(); l++) {
        m_biases.emplace_back(vector::Random(topology[l]));
        m_weights.emplace_back(matrix::Random(
                topology[l], topology[l - 1])); // rows: len(l), cols: len(l-1);
    }
}

vector neural_net::feed_forward(const vector &input) const {
    vector output = input;
    for (unsigned l = 0; l < m_weights.size(); l++) {
        output = m_weights[l] * output + m_biases[l]; // z[l] - weighted sum
        output = output.unaryExpr(&sigmoid);      // a[l] - activation
    }
    return output;
}

void neural_net::propagate_batch(const data_set &training_set,
                                 unsigned start_index, unsigned batch_size,
                                 float learning_rate) {
    std::vector<matrix> summed_delta_weights;
    std::vector<vector> summed_delta_biases;

    // fill sums with 0 matrices and vectors of sizes corresponding to layers in
    // nn
    for (unsigned i = 0; i < m_weights.size(); i++) {
        summed_delta_weights.emplace_back(matrix::Zero(m_weights[i].rows(), m_weights[i].cols()));
        summed_delta_biases.emplace_back(vector::Zero(m_biases[i].rows()));
    }

    // adjust batch size so there is no accessing of elements at indexes larger
    // than possible
    if (start_index + batch_size > training_set.size())
        batch_size = training_set.size() - start_index - 1;

    // starting from start_index get batch_size of in-out pairs from training_set
    // for each in-out pair
    for (unsigned i = start_index; i < start_index + batch_size; i++) {

        // get changes for each layer
        auto d_wb = get_delta_weights_and_biases(
                training_set[i].input, training_set[i].output, learning_rate);
        // add those changes to sums of changes for each layer
        for (unsigned l = 0; l < m_weights.size(); l++) {
            summed_delta_weights[l] += std::get<0>(d_wb)[l];
            summed_delta_biases[l] += std::get<1>(d_wb)[l];
        }
    }

    // after summing up all changes in m_weights and m_biases
    // for each layer in neural net
    for (unsigned l = 0; l < m_weights.size(); l++) {
        // divide summed up changes for that layer by batch_size to get average
        // change
        summed_delta_biases[l] /= batch_size;
        summed_delta_weights[l] /= batch_size;
        // and apply average changes to m_weights and m_biases
        m_biases[l] -= summed_delta_biases[l];
        m_weights[l] -= summed_delta_weights[l];
    }
}

void neural_net::train(const data_set &training_set, unsigned epochs,
                       unsigned batch_size, float learning_rate) {
    // for each epoch
    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        // shuffle training data
        // lazy enough to not implement swap
        // std::random_shuffle( training_set.begin(), training_set.end() );
        // for each batch
        for (unsigned i = 0; i < training_set.size();
             i += batch_size) { // i = start index in training set for batch;
            // propagate
            propagate_batch(training_set, i, batch_size, learning_rate);
        }
    }
}

std::tuple<std::vector<matrix>, std::vector<vector>>
neural_net::get_delta_weights_and_biases(const vector &input,
                                         const vector &desired_output,
                                         float learning_rate) const {

    std::vector<matrix> d_weights;
    d_weights.reserve(m_weights.size());
    std::vector<vector> d_biases;
    d_biases.reserve(m_biases.size());

    // [x] means superscript x
    std::vector<vector> outputs;
    outputs.push_back(input); // output of input layer; // a[l]
    std::vector<vector> sums; // z[l]
    vector output = input;
    for (unsigned l = 0; l < m_weights.size(); l++) {
        output = m_weights[l] * output + m_biases[l]; // z
        sums.push_back(output);
        output = output.unaryExpr(&sigmoid); // a
        outputs.push_back(output);
    }

    // b[L] = (a[L] -y(x)) cwise* d_sigmoid(z[L]);     delta of last layer
    vector err_prev =
            (output - desired_output).cwiseProduct(sums.back().unaryExpr(&d_sigmoid));

    vector gradient_b = err_prev;
    matrix gradient_w =
            err_prev * outputs[outputs.size() - 2]
                    .transpose(); //-1 for index of last -1 for prev layer

    matrix delta_w = gradient_w * learning_rate;
    matrix delta_b = gradient_b * learning_rate;

    d_weights.push_back(delta_w);
    d_biases.emplace_back(delta_b);

    int index_of_last_layer = static_cast<int>(m_weights.size() - 1);
    for (int l = index_of_last_layer - 1; l >= 0;
         l--) { // skip last layer since we did it already;
        // b[l] = (w[l+1]^T * b[l+1]) cwise* d_sigmoid(z[l])
        vector error = (m_weights[l + 1].transpose() * err_prev)
                .cwiseProduct(sums[l].unaryExpr(&d_sigmoid));

        gradient_b = error;
        gradient_w = error * outputs[l].transpose();

        delta_w = gradient_w * learning_rate;
        delta_b = gradient_b * learning_rate;

        d_weights.push_back(delta_w);
        d_biases.emplace_back(delta_b);

        err_prev = error;
    }
    std::reverse(d_weights.begin(), d_weights.end());
    std::reverse(d_biases.begin(), d_biases.end());
    return {d_weights, d_biases};
}

float neural_net::mean_squared_error(const data_set &test_set) const {
    float error_sum = 0;

    for (const auto &in_out_pair : test_set) {
        vector error = in_out_pair.output - feed_forward(in_out_pair.input);
        error_sum += error.squaredNorm();
    }

    return error_sum / float(test_set.size() * 2);
}

void neural_net::serialize(const neural_net &nn, const std::string& file_path) {
    std::ofstream ofs(file_path, std::ios_base::out);

    if(not ofs.is_open()) {
        throw std::runtime_error(std::string{"nnlib: cant open file"} + file_path);
    } else {
        boost::archive::binary_oarchive oa(ofs);
        oa << nn;
    }
    ofs.close();
}

neural_net neural_net::deserialize(const std::string& file_path) {
    neural_net nn;
    std::ifstream ifs(file_path, std::ios_base::in);

    if(not ifs.is_open()) {
        throw std::runtime_error(std::string{"nnlib: cant open file"} + file_path);
    } else {
        boost::archive::binary_iarchive ia(ifs);
        ia >> nn;
    }
    ifs.close();
    return nn;
}

float neural_net::sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

float neural_net::d_sigmoid(float x) {
    float sig_x = sigmoid(x);
    return sig_x * (1 - sig_x);
}

