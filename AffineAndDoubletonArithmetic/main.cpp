#include <iostream>
#include <random>
#include "capd/capdlib.h"
#include "NeuralNetwork.h"
#include "ParseNN.h"

using namespace std;
using namespace capd;

#define LOGGER(x) cout << (#x) << "=" << x << endl;

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

void innDoubletonTest(FullyConnectedNeuralNetwork& nn, DVector x, double e, int N){
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-e,e);
  
  // box centred at x of size [-e,e] in each coordinate
  IVector box = IVector(x) + interval(-e,e);

  // evaluation in doubleton arithmetics
  Doubleton d(box);
  d = nn.eval(d);
  IVector yDltn = d.toVector();
  LOGGER(yDltn);
}

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

void runConvolutionalDoubletonTest(const char* weights, const char* dataset, int dim, int N, double eps, int num, bool addSoftMaxLayer=true){
  FullyConnectedNeuralNetwork nn;
  parseFromFile(nn,weights,addSoftMaxLayer);
  auto pts = readDataSet(dataset,dim);
  for(auto u : pts){
    cout << "#######################\nTest for : " << num << endl;
    innDoubletonTest(nn,u,eps,N);   
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

void FullyConnectedDirectionalDerivative(const char* weights, const char* dataset, const char* directions, int dim, double eps, int num, bool addSoftMaxLayer=true){
  FullyConnectedNeuralNetwork nn;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-eps,eps);

  parseFromFile(nn,weights,addSoftMaxLayer);
  auto pts = readDataSet(dataset,dim);
  auto dirs = readDataSet(directions,dim);

  auto itPts = pts.begin();
  auto itDirs = dirs.begin();

  while(itPts != pts.end() || itDirs != dirs.end()) {
    cout << "#######################\nTest for : " << num << endl;
    IVector mu = IVector(*itPts);
    IVector dir = IVector(*itDirs);
    
    // box centred at x of size [-e,e] in each coordinate
    IVector box = IVector(mu) + dir*interval(-eps,eps);
    // direct evaluation in interval arithmetics
    
    IVector yIntv = nn.eval(box);
    LOGGER(yIntv);
    
    AffineExpr expr;
    AffineFunc e = expr.newVar(interval(-eps,eps)); // zaburzenie jest jedyna zmienna
    AffineVector x = mu+e*dir;
    IVector yAffn = toIVector(nn.eval(x));
    LOGGER(yAffn);
    
    IMatrix A(mu.dimension(),mu.dimension());
    A.column(0) = dir;
    IVector r(mu.dimension());
    r[0] = interval(-eps,eps);
    
    Doubleton d(mu,A,r);
    d = nn.eval(d);
    IVector yDltn = d.toVector();
    LOGGER(yDltn);

    // lower bound on the size of output 
    // computed as the interval hull of NN outpus for N randomly chosen points from the box.
    IVector yRand(nn.eval(mu));
    for(int i=0;i<1000;++i){
      DVector z = *itPts;
      DVector rand_dir = *itDirs;

      for(int j=0;j<x.dimension();++j)
        z[j] += distribution(generator)*rand_dir[j];
      yRand = intervalHull(yRand,IVector(nn.eval(z)));
    }
    LOGGER(yRand);

    if(itPts != pts.end())
    {
        ++itPts;
    }
    if(itDirs != dirs.end())
    {
        ++itDirs;
    } 
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

  std::int8_t num=20;
  double log_start = std::log10(1e-5);
  double log_end = std::log10(1e-2);
  double eps;
  double delta = (log_end - log_start) / (num - 1);

  int start=0;
  int end=19;

  for (int i = start; i <= end; ++i) {
      eps = std::pow(10, log_start + i * delta);

      // cout.sync_with_stdio(false);
      // runConvolutionalTest("./data/cifar/cnn_small_eps_0_001.txt",strides,"./data/cifar/the_most_uncertain_points_eps_0_001.txt",1000,eps,i,true);
      runFullyConnectedTest("./data/digits/mlp_eps_0_05.txt", "./data/digits/the_most_uncertain_points_eps_0_05.txt", 64, 1000, eps, i, true);
      // runConvolutionalDoubletonTest("./data/mnist/cnn_small_eps_0_01_toeplitz.txt", "./data/mnist/the_most_uncertain_points_eps_0_01_toeplitz.txt", 784, 1000, eps, i, true);

  }
}
