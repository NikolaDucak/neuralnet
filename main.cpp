#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include "neural_net.h"

static const std::string usage =
        R"(
nncli file_name <command> <arguments>

commands:
    make:  generates a neural net & serializes it to file_name
        argument: m_topology of neural net in format "num;num;num"
        eg. nncli make net.nn 1-2-3-4

    train: deserializes network, trains it & serializes it
        arguments:
            1) path to training set
            2) epochs (integer)
            3) batch size (integer)
            4) learning rate (decimal)
        eg. nncli net.nn train ../path/to/dataset 1000 100 2.5
        dataset format: |-----input------|-output-|
                        0.53, 0.012, 0.99, 0, 1

    feed: deserializes network in file_name, propagates input
        argument: input vector in format "num-num-num"
        eg. nncli net.nn feed 0.53-0.61-1.0
)";

// utilities
template<typename numeric_t>
static std::vector<numeric_t> parse_vector(std::string_view vector_str);
static data_set parse_train_set(const std::string &filename, int input_size, int output_size);

// actions
static void make(const std::string &file_name, const std::string &topology_str);
static void train(const std::string &file_name, const char* argv[]);
static void feed(const std::string &file_name, const std::string &input_vector_str);

int main(int argc, const char *argv[]) {
    if (argc == 2 && !std::strcmp(argv[1], "help")) {
        std::cout << usage << std::endl;
        return 0;
    } else if (argc < 4){
        std::cout << "Oops! Bad number of arguments! \n"
                  << "See `nncli help'." << std::endl;
        return 1;
    }

    if (not std::strcmp(argv[2], "make")) {
        make(argv[1], argv[3]);
    } else if (not std::strcmp(argv[2], "train")) {
        if (argc != 7) {
            std::cout << "Bad number of arguments for 'train' command." << std::endl;
        } else {
            train(argv[1], argv+3);
        }
    } else if (not std::strcmp(argv[2], "feed")) {
        feed(argv[1], argv[3]);
    } else {
        std::cout << "Unknown action: " << argv[2] << ". See 'nncli help'." << std::endl;
    }
}

template<typename numeric_t>
std::vector<numeric_t> parse_vector(std::string_view vector_str) {
    std::vector<std::string> tokens;
    std::vector<numeric_t> result;

    boost::split(tokens, vector_str, boost::is_any_of("-"));

    std::transform(tokens.begin(), tokens.end(),
                   std::back_inserter(result),
                   [](const auto &s) { return boost::lexical_cast<numeric_t>(s); });

    return result;
}

static data_set parse_train_set(const std::string &filename, int input_size, int output_size) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Can't open file!");

    data_set set;

    for (std::string line; std::getline(file, line);) {
        std::stringstream line_stream(line);

        vector input(input_size);
        vector output(output_size);

        for (int i = 0; i < input_size; i++) {
            line_stream >> input[i];
            if (line_stream.peek() == ',') line_stream.ignore();
        }

        for (int i = 0; i < output_size; i++) {
            line_stream >> output[i];
            if (line_stream.peek() == ',') line_stream.ignore();
        }

        set.push_back({input, output});
    }

    return set;
}

void make(const std::string &file_name, const std::string &topology_str) {
    neural_net nn{parse_vector<unsigned int>(topology_str)};
    neural_net::serialize(nn, std::string(file_name));
    std::cout << "Created: " << file_name
              << " with m_topology " << topology_str
              << std::endl;
}

void train(const std::string &file_name, const char* argv[]) {
    auto epochs = std::stoi(argv[1]);
    auto batch = std::stoi(argv[2]);
    auto learning_rate = std::stof(argv[3]);
    neural_net nn = neural_net::deserialize(file_name);
    auto train_set = parse_train_set(argv[0],
                                     nn.get_topology().front(),
                                     nn.get_topology().back());

    std::cout << "\nStarting training with:\n"
              << "\tnet: " << file_name << '\n'
              << "\tdataset: " << argv[0] << '\n'
              << "\tepochs: " << epochs << '\n'
              << "\tbatch size: " << batch << '\n'
              << "\tlearning rate: " << learning_rate << '\n'
              << "\ttraining set size:" << train_set.size() << '\n'
              << "\t...." << std::endl;

    nn.train(train_set, epochs, batch, learning_rate);

    std::cout << "Finished!\n" << std::endl;

    neural_net::serialize(nn, file_name);
}

void feed(const std::string &file_name, const std::string &input_vector_str) {
    neural_net nn = neural_net::deserialize(file_name);

    auto std_vec = parse_vector<float>(input_vector_str);
    auto input_vector = Eigen::Map<vector>(std_vec.data(), std_vec.size());

    if (input_vector.size() != nn.get_topology().front()) {
        std::cout << "input vector '" << input_vector_str
                  << "' is of size " << input_vector.size() << " but '" << file_name
                  << "' takes input vector of size " << nn.get_topology().front()
                  << std::endl;
        exit(1);
    }

    auto output_vector = nn.feed_forward(input_vector);

    static const std::string delimiter = " | ";
    std::stringstream ss;
    std::copy(output_vector.data(), output_vector.data() + output_vector.size(),
              std::ostream_iterator<float>(ss, delimiter.c_str()));
    auto output_str = ss.str().substr(0, (ss.str().size()) - delimiter.size());
    std::cout << output_str << std::endl;
}
