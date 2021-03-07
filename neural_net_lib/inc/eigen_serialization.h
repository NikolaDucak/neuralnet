#ifndef __NEURAL_NETWORK_EIGEN_SERIALIZATION_H__
#define __NEURAL_NETWORK_EIGEN_SERIALIZATION_H__

friend class boost::serialization::access;

// straight up ripped of from:
// https://stackoverflow.com/questions/18382457/eigen-and-boostserialize

template <class Archive>
void save(Archive &ar, const unsigned int version) const {
    derived().eval();
    const Index rows = derived().rows(), cols = derived().cols();
    ar & rows;
    ar & cols;
    for (Index j = 0; j < cols; ++j)
        for (Index i = 0; i < rows; ++i)
            ar & derived().coeff(i, j);
}

template <class Archive>
void load(Archive &ar, const unsigned int version) {
    Index rows, cols;
    ar & rows;
    ar & cols;
    if (rows != derived().rows() || cols != derived().cols())
        derived().resize(rows, cols);
    ar & boost::serialization::make_array(derived().data(), derived().size());
}

template <class Archive>
void serialize(Archive &ar, const unsigned int file_version) {
    boost::serialization::split_member(ar, *this, file_version);
}

#endif // __NEURAL_NETWORK_EIGEN_SERIALIZATION_H__
