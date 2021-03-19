# Neural network implementation in C++
A simple feed forward neural net with stochastic gradient descent as learning
method. Written in an effort to better understand innerworkings of neural
networks. That adventure was made easier thanks to [Neural Networks and Deep Learing](http://neuralnetworksanddeeplearning.com/) e-book.
Also,for efficient matrix multiplication [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
linear algebra library was used.

## Compiling
**Dependancies**: since Eigen is included within the source code, only `boost` in necessary.

Clone and navigate to root directory of the repo, then:
```shell
mkdir build
cd build
cmake ..
make
```
After that, `nncli` executable will be generated in the same directory.

## Usage
Usage is centered around serialized neural network files. `nncli`
provides commands to generate, train and feed input to serialized networks.
Run `nncli help` for command line details.

**Example**
```shell
$ nncli test.nn make 2-3-1

Created test.nn with 2-3-1 m_topology.

$ nncli test.nn train example_dataset/xor.tre 1000 100 2.5

Starting training with:
        net: net.nn
        dataset: ../example_dataset/xor.tre
        epochs: 1000
        batch size: 100
        learning rate: 2.5
        training set size:4
        ....
Finished!

$ nncli test.nn feed 1-1
0.028531

```