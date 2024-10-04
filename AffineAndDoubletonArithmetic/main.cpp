/////////////////////////////////////////////////////////////////////////////
/// @file main.cpp

/// @author (C) 2024 Daniel Wilczak
///
/// This file is distributed under the terms of the GNU General Public License.

/// This file contains implementation of tests, which compare bounds 
/// on a neural network output computed by means of LB, IBP, DA and AA.

#include <iostream>
#include <random>
#include "capd/capdlib.h"
#include "NeuralNetwork.h"
#include "ParseNN.h"

using namespace std;
using namespace capd;

#define LOGGER(x) cout << (#x) << "=" << x << endl;

/// Computation of a bound on a neural netowrk output using all four methods: IBP, DA, AA and LB.
/// Also LB is computed. 
/// @param nn[in] - a fully connected neural network
/// @param x[in] - a point, which is the centre of a box
/// @param e[in] - size of the box centred at x
/// @param N[in] - number of random points for LB method
void innTest(FullyConnectedNeuralNetwork& nn, DVector x, double e, int N){
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-e,e);
  
  // box centred at x of size [-e,e] in each coordinate
  IVector box = IVector(x) + interval(-e,e);

  // direct evaluation in interval arithmetics
  IVector yIntv = nn.eval(box);
  LOGGER(yIntv);

  // evaluation in doubleton arithmetics
  Doubleton d(box);
  d = nn.eval(d);
  IVector yDltn = d.toVector();
  LOGGER(yDltn);

  // evaluation in affine arithmetics
  AffineExpr expr;
  AffineVector v = expr.newVector(box);
  IVector yAffn = toIVector(nn.eval(v));
  LOGGER(yAffn);

  // lower bound on the size of output 
  // computed as the interval hull of NN outpus for N randomly chosen points from the box.
  IVector yRand(nn.eval(x));
  for(int i=0;i<N;++i){
    DVector z = x;
    for(int j=0;j<x.dimension();++j)
      z[j] += distribution(generator);
    yRand = intervalHull(yRand,IVector(nn.eval(z)));
  }
  LOGGER(yRand);
}

/// Computation of a bound on a fully connected neural netowrk output using DA.
/// @param nn[in] - a fully connected neural network
/// @param x[in] - a point, which is the centre of a box
/// @param e[in] - size of the box centred at x
void innDoubletonTest(FullyConnectedNeuralNetwork& nn, DVector x, double e){
  // box centred at x of size [-e,e] in each coordinate
  IVector box = IVector(x) + interval(-e,e);

  // evaluation in doubleton arithmetics
  Doubleton d(box);
  d = nn.eval(d);
  IVector yDltn = d.toVector();
  LOGGER(yDltn);
}
/// Computation of a bound on a convolutional neural netowrk output using: IBP, AA and LB.
/// Also LB is computed. 
/// @param nn[in] - a convolutional neural network
/// @param x[in] - a point, which is the centre of a box
/// @param e[in] - size of the box centred at x
/// @param N[in] - number of random points for LB method
void innTest(ConvolutionalNeuralNetwork& nn, DTensor x, double e, int N){
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-e,e);
  
  // box centred at x of size [-e,e] in each coordinate
  ITensor box = convert(x) + interval(-e,e);
  // direct evaluation in interval arithmetics
  auto yIntv = nn.eval(box);
  LOGGER(yIntv);

  // evaluation in affine arithmetics
  AffineExpr expr;
  ATensor v = expr.newTensor(box);
  auto yAffn = toIVector(nn.eval(v));
  LOGGER(yAffn);

  // lower bound on the size of output 
  // computed as the interval hull of NN outpus for N randomly chosen points from the box.
  auto yRand = convert(nn.eval(x));
  for(int i=0;i<N;++i){
    DTensor  z = x;
    for(auto& i : z)
      for(auto& j : i)        
        for(auto k=j.begin();k!=j.end();++k)
          *k += distribution(generator);
    yRand = intervalHull(yRand,convert(nn.eval(z)));
  }
  LOGGER(yRand);
}


void runFullyConnectedTest(const char* weights, const char* dataset, int dim, int N, double eps, int num, bool addSoftMaxLayer=true){
  FullyConnectedNeuralNetwork nn;
  parseFromFile(nn,weights,addSoftMaxLayer);
  auto pts = readDataSet(dataset,dim);
  for(auto u : pts){
    cout << "#######################\nTest for : " << num << endl;
    innTest(nn,u,eps,N);   
  }
}

void runConvolutionalDoubletonTest(const char* weights, const char* dataset, int dim, double eps, int num, bool addSoftMaxLayer=true){
  FullyConnectedNeuralNetwork nn;
  parseFromFile(nn,weights,addSoftMaxLayer);
  auto pts = readDataSet(dataset,dim);
  for(auto u : pts){
    cout << "#######################\nTest for : " << num << endl;
    innDoubletonTest(nn,u,eps);   
  }
}

void runConvolutionalTest(const char* weights, const std::map<int,int>& strides, const char* dataset, int N, double eps, int num, bool addSoftMaxLayer){
  ConvolutionalNeuralNetwork nn;
  parseFromFile(nn,weights,addSoftMaxLayer);
  
  for(auto i : strides)
    nn.getLayer(i.first)->setStride(i.second);
  
  auto pts = readDataSet(dataset);
 
  for(auto u : pts){
    cout << "#######################\nTest for : " << num << endl;
    innTest(nn,u,eps,N);   
  }
}

int main(){
  cout.precision(4);
  
  //~ runFullyConnectedTest("./data/neural_net_weights.txt","./data/dataset.txt",2,10000,1e-1,false);
  //~ runFullyConnectedTest("./data/digits_neural_net_weights.txt","./data/digits_dataset.txt",64,10000,1e-1,true);
  //~ runFullyConnectedTest("./data/MNIST_neural_net_weights.txt","./data/MNIST_dataset.txt",784,10000,1e-3,true);
  

  std::map<int,int> strides;
  strides[0] = 2; strides[2] = 1;
  // strides[0] = 1; strides[2] = 2; strides[4] = 1; strides[6] = 2;
  // strides[0] = 1; strides[2] = 1; strides[4] = 2; strides[6] = 1; strides[8] = 1;

  int num=20;
  double log_start = std::log10(1e-5);
  double log_end = std::log10(1e-2);
  double delta = (log_end - log_start) / (num - 1);

  int start=0;
  int end=19;

  for (int i = start; i <= end; ++i) {
      double eps = std::pow(10, log_start + i * delta);

      // cout.sync_with_stdio(false);
      // runConvolutionalTest("./data/cifar/cnn_small_eps_0_001.txt",strides,"./data/cifar/the_most_uncertain_points_eps_0_001.txt",1000,eps,i,true);
      runFullyConnectedTest("./data/digits/mlp_eps_0_05.txt", "./data/digits/the_most_uncertain_points_eps_0_05.txt", 64, 1000, eps, i, true);
      // runConvolutionalDoubletonTest("./data/mnist/cnn_small_eps_0_01_toeplitz.txt", "./data/mnist/the_most_uncertain_points_eps_0_01_toeplitz.txt", 784, eps, i, true);

  }
}
