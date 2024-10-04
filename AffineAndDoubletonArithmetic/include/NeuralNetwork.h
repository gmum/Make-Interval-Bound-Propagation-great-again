/////////////////////////////////////////////////////////////////////////////
/// @file NeuralNetwork.h
///
/// @author (C) 2024 Daniel Wilczak
///
/// This file is distributed under the terms of the GNU General Public License.

#ifndef __CAPD_NEURAL_NETWORK_LAYER__
#define __CAPD_NEURAL_NETWORK_LAYER__

#include "AffineFunc.h"
#include "Doubleton.h"
#include "softmax.h"
#include "relu.h"

/// Abstract interfeace of a layer in a neural network. 
/// Inherited classes should implement evaluation of a layer for various 
/// types of subsets of R^n, especially interval vectors, Affine vectors, 
/// doubletons and tensors.
struct NeuralNetworkLayer{
  virtual AffineVector eval(const AffineVector&) const = 0;
  virtual Doubleton eval(const Doubleton&) const = 0;
  virtual capd::IVector eval(const capd::IVector& x) const = 0;
  virtual capd::DVector eval(const capd::DVector& x) const = 0;

  virtual DTensor eval(const DTensor& in) const = 0;
  virtual ITensor eval(const ITensor& in) const = 0;
  virtual ATensor eval(const ATensor& in) const = 0;
  
  virtual void setStride(int s){}
};


/// Implementation of affine layer
struct AffineLayer : public NeuralNetworkLayer{
  AffineLayer(const capd::IMatrix& A, const capd::IVector& b) 
    : iA(A), ib(b),  
      dA(capd::vectalg::convertObject<capd::DMatrix>(A)),
      db(capd::vectalg::convertObject<capd::DVector>(b))
  {}
  AffineVector eval(const AffineVector& x) const { return iA*x+ib; }
  Doubleton eval(const Doubleton& x) const { return x.affineTransformation(iA,ib); }
  capd::IVector eval(const capd::IVector& x) const { return iA*x+ib; }
  capd::DVector eval(const capd::DVector& x) const { return dA*x+db; }

  DTensor eval(const DTensor& in) const { throw std::logic_error("AffineLayer::eval(const DTensor& in)"); }
  ITensor eval(const ITensor& in) const { throw std::logic_error("AffineLayer::eval(const ITensor& in)"); }
  ATensor eval(const ATensor& in) const { throw std::logic_error("AffineLayer::eval(const ATensor& in)"); }

  capd::IMatrix iA;
  capd::IVector ib;
  capd::DMatrix dA;
  capd::DVector db;
};

/// Implementation of ReLU layer
struct ReluLayer : public NeuralNetworkLayer{
  AffineVector eval(const AffineVector& x) const { return relu(x); }
  Doubleton eval(const Doubleton& x) const { return relu(x);}
  capd::IVector eval(const capd::IVector& x) const { return relu(x); }
  capd::DVector eval(const capd::DVector& x) const { return relu(x); }

  DTensor eval(const DTensor& in) const { return evalTensor(in); }
  ITensor eval(const ITensor& in) const { return evalTensor(in); }
  ATensor eval(const ATensor& in) const { return evalTensor(in); }
  
  template<class Tensor>
  static Tensor evalTensor(Tensor t){
    for(unsigned i=0;i<t.size();++i)
      for(unsigned j=0;j<t[i].size();++j)
        for(unsigned k=0;k<t[i][j].numberOfRows();++k)
          for(unsigned s=0;s<t[i][j].numberOfColumns();++s)
            t[i][j][k][s] = relu(t[i][j][k][s]);
    return t;
  }
};

/// Implementation of softmax layer
struct SoftMaxLayer : public NeuralNetworkLayer{
  AffineVector eval(const AffineVector& x) const { return softmax(x); }
  Doubleton eval(const Doubleton& x) const {  return softmax(x); }
  capd::IVector eval(const capd::IVector& x) const { return softmax(x);}
  capd::DVector eval(const capd::DVector& x) const { return softmax(x);}

  DTensor eval(const DTensor& in) const { throw std::logic_error("SoftMaxLayer::eval(const DTensor& in)"); }
  ITensor eval(const ITensor& in) const { throw std::logic_error("SoftMaxLayer::eval(const ITensor& in)"); }
  ATensor eval(const ATensor& in) const { throw std::logic_error("SoftMaxLayer::eval(const ATensor& in)"); }
};

/// Implementation of convolutional layer
struct ConvolutionalLayer : public NeuralNetworkLayer{  
  ConvolutionalLayer(const DTensor& t, const capd::DVector& bias) 
    : dWeights(t), dBias(bias), stride(1)
  {
    iWeights = convert(dWeights);
    iBias = capd::vectalg::convertObject<capd::IVector>(dBias);
  }
  
  AffineVector eval(const AffineVector&) const { throw std::logic_error("ConvolutionalLayer::eval(const AffineVector&)"); }
  Doubleton eval(const Doubleton&) const { throw std::logic_error("ConvolutionalLayer::eval(const Doubleton&)"); }
  capd::IVector eval(const capd::IVector& x) const { throw std::logic_error("ConvolutionalLayer::eval(const IVector&)"); }
  capd::DVector eval(const capd::DVector& x) const { throw std::logic_error("ConvolutionalLayer::eval(const DVector&)"); }

  DTensor eval(const DTensor& in) const{
    return eval(dWeights,dBias,in,stride);
  }

  ITensor eval(const ITensor& in) const{
    return eval(iWeights,iBias,in,stride);
  }

  ATensor eval(const ATensor& in) const{
    return eval(iWeights,iBias,in,stride);
  }

  template<class Tensor, class Bias, class Input>
  static Input eval(const Tensor& weights, const Bias& bias, const Input& in, int stride) {
    const int kh = weights[0][0].numberOfRows();  // Kernel height
    const int kw = weights[0][0].numberOfColumns();  // Kernel width
    const int h = 1 + (in[0][0].numberOfRows() - kh) / stride;  // Output height
    const int w = 1 + (in[0][0].numberOfColumns() - kw) / stride;  // Output width

    Input result;

    for (unsigned i = 0; i < in.size(); ++i) { 
        typename Input::value_type b; 
        for (unsigned j = 0; j < bias.dimension(); ++j) { 
            typename Input::value_type::value_type M(h, w);
            for (int r = 0; r < h; ++r) {
                for (int s = 0; s < w; ++s) {
                    for (unsigned k = 0; k < in[0].size(); ++k) { 
                        for (int n = 0; n < kh; ++n) {
                            for (int m = 0; m < kw; ++m) {
                                M[r][s] += weights[j][k][n][m] * in[i][k][r * stride + n][s * stride + m];
                            }
                        }
                    }
                    // Add bias after the convolution is applied for the entire channel
                    M[r][s] = M[r][s] + bias[j];
                }
            }
          b.push_back(M);
        }
        result.push_back(b);
    }
    return result;
}

  void setStride(int s) { stride = s; }
  
  ITensor iWeights;
  capd::IVector iBias;
  DTensor dWeights;
  capd::DVector dBias;
  int stride;
};

/// A data structure that stores layers of a neural network
struct NeuralNetwork{
public:
  void add(NeuralNetworkLayer* layer){
    layers.push_back(layer);
  }
  
  ~NeuralNetwork(){
    for(auto layer : layers) delete layer;
  }
  
  NeuralNetworkLayer* getLayer(int i) { return layers[i]; }
  
  std::vector<NeuralNetworkLayer*> layers;
};

/// Implementation of fully connected neural network.
/// The class provides a generic method for evaluation of a neural network 
/// for different subsets of R^n, such as interval vectors, doubletons, affine sets.
struct FullyConnectedNeuralNetwork : public NeuralNetwork{
public:
  template<class T>
  T eval(T x){
    for(auto layer : layers)
      x = layer->eval(x);
    return x;
  }
};

/// An auxiliary class that flattens tensor to a vector
struct Flatten{
  static std::vector<capd::DVector> flatten(const DTensor& t) { return f<capd::DVector>(t); }
  static std::vector<capd::IVector> flatten(const ITensor& t) { return f<capd::IVector>(t); }
  static std::vector<AffineVector> flatten(const ATensor& t)  { return f<AffineVector>(t); }
  
private:
  template<class Result, class Tensor>
  static std::vector<Result> f(const Tensor& x){
    std::vector<Result> result;
    for(unsigned int i=0;i<x.size();++i){
      Result r(x[i][0].numberOfRows()*x[i][0].numberOfColumns()*x[i].size());
      unsigned int s=0;
      for(int j=0;j<x[i].size();++j)
        for(auto n=x[i][j].begin();n!=x[i][j].end();++n,++s)
          r[s] = *n;
      result.push_back(r);
    }
    return result;
  }  
};


/// Implementation of convolutional  neural network.
/// The class provides a generic method for evaluation of a neural network 
/// for different subsets of R^n, such as interval vectors, doubletons, affine sets.
struct ConvolutionalNeuralNetwork : public NeuralNetwork{
  template<class T>
  auto eval(T x) -> decltype(Flatten::flatten(x)){
    unsigned i;
    for(i=0;i<layers.size();++i){
      if(dynamic_cast<AffineLayer*>(layers[i])) break;
      x = layers[i]->eval(x);
    }
    auto y = Flatten::flatten(x);
    for(;i<layers.size();++i){
      for(auto& z : y)
        z = layers[i]->eval(z);
    }
    return y;
  }
};

#endif
