#ifndef __DW_RELU_H__
#define __DW_RELU_H__

#include "capd/capdlib.h"

template<class Scalar>
capd::vectalg::Vector <Scalar,0> relu(capd::vectalg::Vector <Scalar,0> u){
  auto r = u;
  for(int i=0;i<r.dimension();++i) 
    r[i] = capd::max(Scalar(0.),r[i]);
  return r;
}

inline double relu(double x) { return capd::max(0.,x); }
inline capd::interval relu(capd::interval x) { return capd::max(capd::interval(0.),x); }

#endif