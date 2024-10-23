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

int main(int argc, char *argv[]) {
    std::cout << "argc: " << argc << "\n";
    for(int i = 0; i < argc; ++i){
        std::cout << "argv[" << i << "]: " << argv[i] << "\n";
    }

    std::string function_name = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];

    double log_start = 0.0, log_end = 0.0;
    try {
        log_start = std::log10(std::stod(argv[4]));  
        log_end = std::log10(std::stod(argv[5])); 
    } catch(const std::invalid_argument&) {
        std::cerr << "Error: Invalid log_start or log_end value.\n";
        return 1;
    } catch(const std::out_of_range&) {
        std::cerr << "Error: log_start or log_end value out of range.\n";
        return 1;
    }

    std::map<int,int> strides;
    int input_size = 0;

    if (function_name == "runFullyConnectedTest" || function_name == "runConvolutionalDoubletonTest") {
        input_size = std::stoi(argv[6]);
    } else {
        std::string cnn_arch_type = argv[6];

        if (cnn_arch_type == "cnn_small") {
            strides[0] = 2; strides[2] = 1;
        } else if (cnn_arch_type == "cnn_medium") {
            strides[0] = 1; strides[2] = 2; strides[4] = 1; strides[6] = 2;
        } else if (cnn_arch_type == "cnn_large") {
            strides[0] = 1; strides[2] = 1; strides[4] = 2; strides[6] = 1; strides[8] = 1;
        }
    }

    std::cout.precision(4);
    const std::int8_t num = 20;
    double delta = (log_end - log_start) / (num - 1);
    const int start = 0;
    const int end = num - 1;

    for(int i = start; i <= end; ++i){
        double eps = std::pow(10, log_start + i * delta);

        if(function_name == "runFullyConnectedTest"){
            runFullyConnectedTest(input_path.c_str(), output_path.c_str(), input_size, 1000, eps, i, true);
        }
        else if(function_name == "runConvolutionalTest"){
            runConvolutionalTest(input_path.c_str(), strides, output_path.c_str(), 1000, eps, i, true);
        }
        else if(function_name == "runConvolutionalDoubletonTest"){
            runConvolutionalDoubletonTest(input_path.c_str(), output_path.c_str(), input_size, eps, i, true);
        }
        else{
            std::cerr << "Error: Unknown function '" << function_name << "'\n";
            return 1;
        }
    }

    return 0;
}
