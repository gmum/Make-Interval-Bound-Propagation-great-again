#ifndef __DW_DOUBLETON__
#define __DW_DOUBLETON__
#include "capd/capdlib.h"

class Doubleton{
public:
  Doubleton(const capd::IVector& x)
    : x(x), r(x.dimension()), q(x.dimension()),
      C(capd::IMatrix::Identity(x.dimension())), Q(capd::IMatrix::Identity(x.dimension()))
  {
      split(this->x,this->x,this->r);
  }

  Doubleton(const capd::IVector& x,const capd::IMatrix& C, const capd::IVector& r,const capd::IMatrix& Q, const capd::IVector& q)
    : x(x), r(r), q(q), C(C), Q(Q)
  {}
  
  Doubleton(const capd::IVector& x,const capd::IMatrix& C, capd::IVector r)
    : x(x), r(r), q(x.dimension()),
      C(C), Q(capd::IMatrix::Identity(x.dimension()))
  {}

  Doubleton affineTransformation(const capd::IMatrix& A, const capd::IVector& b) const{
    return Doubleton(A*x+b,A*C,r,A*Q,q);
  }
  
  capd::IVector toVector() const { return x+C*r+Q*q; }

  capd::IVector x, r, q;
  capd::IMatrix C, Q;
};

Doubleton relu(Doubleton);
Doubleton softmax(const Doubleton&);

#endif