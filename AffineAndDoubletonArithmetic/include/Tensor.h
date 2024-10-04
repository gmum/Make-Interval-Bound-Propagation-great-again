#include <vector>

#ifndef __CAPD_TENSOR_H__
#define __CAPD_TENSOR_H__

typedef std::vector<std::vector<capd::DMatrix>> DTensor;
typedef std::vector<std::vector<capd::IMatrix>> ITensor;

inline ITensor convert(const DTensor& t){
  ITensor r;
  for(const auto& i : t){
    ITensor::value_type b;
    for(const auto& m : i)
      b.push_back(capd::IMatrix(m));
    r.push_back(b);
  }
  return r;
}

inline std::vector<capd::IVector> convert(const std::vector<capd::DVector>& u){
  std::vector<capd::IVector> r;
  for(const auto& x : u)
    r.push_back(capd::IVector(x));
  return r;
}

inline ITensor operator+(ITensor t, capd::interval e){
  for(auto& i : t)
    for(auto& j : i)
      j += e;
  return t;
}

inline std::vector<capd::IVector> intervalHull(const std::vector<capd::IVector>& a, std::vector<capd::IVector> b){
  for(unsigned i=0;i<b.size();++i)
      b[i] = intervalHull(b[i],a[i]);
  return b;
}

#endif