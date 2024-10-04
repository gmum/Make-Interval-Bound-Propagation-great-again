#ifndef __DW_SOFTMAX_H__
#define __DW_SOFTMAX_H__

#include "capd/capdlib.h"

template<class Scalar>
capd::vectalg::Vector <Scalar,0> softmax(capd::vectalg::Vector <Scalar,0> u){
  // take max of components
  double C = 0.;

  for(unsigned i=0;i<u.dimension();++i)
    C = capd::max(C,rightBound(u[i]));
  Scalar s = 0.;

  for(unsigned i=0;i<u.dimension();++i){
    u[i] = exp(u[i]-C);
    s += u[i];
  }

  for(unsigned i=0;i<u.dimension();++i) {
    if (isSingular(s)) {    
      u[i] = capd::min(s,Scalar(1.));
    } else {
    u[i] = capd::min(u[i]/s,Scalar(1.));
    }
  }
  return u;
}

#endif