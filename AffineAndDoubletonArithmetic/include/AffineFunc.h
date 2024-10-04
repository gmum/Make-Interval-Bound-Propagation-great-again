/////////////////////////////////////////////////////////////////////////////
/// @file AffineFunc.h
///
/// @author (C) 2024 Daniel Wilczak
///
/// This file is distributed under the terms of the GNU General Public License.

#ifndef __CAPD_AFFINE_FUNC__
#define __CAPD_AFFINE_FUNC__

#include <set>
#include <mutex>
#include <map>
#include <cmath>
#include <ostream>
#include "capd/capdlib.h"
#include "Tensor.h"

class AffineExpr;

/**
 * class AffineFunc implements basic rules of affine arithmetics.
 * Implementation is thread safe.
 * AffineFunc represents an affine function with independent variables registered 
 * in expr member. 
 * If expr==nullptr, the object behaves like regular interval.
 * 
 */
class AffineFunc{
  public:
    AffineFunc(){ coeffs[0] = 0.; }
    explicit AffineFunc(const capd::interval& c){ coeffs[0] = c; }

    friend class AffineExpr;
    
    friend std::ostream& operator<<(std::ostream&, const AffineFunc&);
    
    friend AffineFunc operator+(const capd::interval&, const AffineFunc&);
    friend AffineFunc operator+(const AffineFunc& f ,const capd::interval& c);

    friend AffineFunc operator-(const capd::interval&, const AffineFunc&);
    friend AffineFunc operator-(const AffineFunc&, const capd::interval&);

    friend AffineFunc operator*(const capd::interval&, const AffineFunc&);
    friend AffineFunc operator*(const AffineFunc& f ,const capd::interval& c);

    friend AffineFunc operator+(const AffineFunc&, const AffineFunc&);
    friend AffineFunc operator-(const AffineFunc&, const AffineFunc&);
    friend AffineFunc operator*(const AffineFunc&, const AffineFunc&);

    friend AffineFunc relu(const AffineFunc& f);
    friend capd::vectalg::Vector<AffineFunc,0> softmax(capd::vectalg::Vector<AffineFunc,0> z);

    typedef std::map<int,capd::interval> CoeffsType;
    AffineFunc& operator+=(const AffineFunc&);
    
    capd::interval toInterval() const;
    void addVar(const capd::interval& c);
  private:

    AffineFunc(AffineExpr* expr) : expr(expr) {}
    AffineExpr* expr = nullptr;
    CoeffsType coeffs;
    static const capd::interval I;
    
    static AffineExpr* checkExpr(const AffineFunc& f, const AffineFunc& g){
      if(f.expr==g.expr) 
        return f.expr;
      if(f.expr==nullptr)
        return g.expr;
      if(g.expr==nullptr)
        return f.expr;
      throw std::runtime_error("AffineFunc::testAndSetExpr - incompatible expressions");
    }
};

typedef capd::vectalg::Vector<AffineFunc,0> AffineVector;
typedef capd::vectalg::Matrix<AffineFunc,0,0> AffineMatrix;
typedef std::vector<std::vector<AffineMatrix>> ATensor;

inline capd::IVector toIVector(const AffineVector& a){
    capd::IVector result(a.dimension());
    for(int i=0;i<a.dimension();++i)
      result[i] = a[i].toInterval();
    return result;
}

inline std::vector<capd::IVector> toIVector(const std::vector<AffineVector>& a){
    std::vector<capd::IVector> result;
    for(const auto& x : a)
      result.push_back(toIVector(x));    
    return result;
}

class AffineExpr{ 
  public:
    AffineExpr(){}
    
    void addVar(AffineFunc& f, capd::interval c){
      capd::interval q;
      f.coeffs[0] += c;
      f.coeffs[0].split(q);
      double r = capd::max(fabs(q.leftBound()),q.rightBound());
      
      const std::lock_guard<std::mutex> lock(m_mutex);
      f.coeffs[++current] = r;
    }

    AffineFunc newVar(capd::interval c){
      AffineFunc result(this);
      addVar(result,c);
      return result;
    }
    
    AffineVector newVector(capd::IVector u){
      AffineVector result(u.dimension(),false);
      for(int i=0;i<u.dimension();++i){
        result[i].expr = this;
        addVar(result[i],u[i]);
      }
      return result;
    }
    
    AffineMatrix newMatrix(capd::IMatrix u){
      AffineMatrix result(u.dimension(),false);
      for(int i=0;i<u.numberOfRows();++i){
        for(int j=0;j<u.numberOfColumns();++j){
          result[i][j].expr = this;
          addVar(result[i][j],u[i][j]);
        }
      }
      return result;
    }
    
    ATensor newTensor(ITensor u){
      ATensor result;
      for(auto& i : u){
        ATensor::value_type b;
        for(auto& j : i)
          b.push_back(newMatrix(j));
        result.push_back(b);
      }
      return result;
    }

    AffineFunc newConst(capd::interval c){
      AffineFunc result(this);
      result.coeffs[0] = c;
      return result;
    }
  private:
    int current = 0;
    std::mutex m_mutex;
};

inline AffineVector operator*(const capd::IMatrix A, const AffineVector& u){
  return capd::vectalg::matrixByVector<AffineVector>(A,u);
}

inline AffineVector operator*(const AffineFunc& c, const capd::IVector& u){
  AffineVector result(u.dimension());
  for(unsigned i=0;i<u.dimension();++i)
    result[i] = c*u[i];
  return result;
}

inline AffineMatrix operator*(const AffineFunc& c, const capd::IMatrix& u){
  AffineMatrix result(u.dimension());
  for(unsigned i=0;i<u.numberOfRows();++i)
    for(unsigned j=0;j<u.numberOfColumns();++j)
      result[i][j] = u[i][j]*c;
  return result;
}

inline AffineVector operator+(const AffineVector& u,const capd::IVector& v){
  return capd::vectalg::addObjects<AffineVector>(u,v);
}

inline AffineVector operator+(const capd::IVector& v,const AffineVector& u){
  return capd::vectalg::addObjects<AffineVector>(u,v);
}

inline AffineMatrix operator+(const AffineMatrix& u,const capd::IMatrix& v){
  return capd::vectalg::addObjects<AffineMatrix>(u,v);
}

inline AffineMatrix operator+(const capd::IMatrix& v, const AffineMatrix& u){
  return capd::vectalg::addObjects<AffineMatrix>(u,v);
}

inline AffineVector operator-(const AffineVector& u,const capd::IVector& v){
  return capd::vectalg::subtractObjects<AffineVector>(u,v);
}

inline ATensor operator*(const AffineFunc& c, const ITensor& x){
  ATensor result;
  for(unsigned i=0;i<x.size();++i){
    ATensor::value_type b;
    for(unsigned j=0;j<x[i].size();++j)
      b.push_back(c*x[i][j]);
    result.push_back(b);
  }
  return result;
}

inline ATensor operator+(const ITensor& c, ATensor x){
  for(unsigned i=0;i<x.size();++i)
    for(unsigned j=0;j<x[i].size();++j)
      x[i][j] = x[i][j] + c[i][j];
  return x;
}

AffineVector relu(const AffineVector&);
AffineVector softmax(AffineVector z); 

#endif
