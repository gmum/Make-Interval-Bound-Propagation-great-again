#include "AffineFunc.h"
#include "capd/vectalg/Vector.hpp"
#include "capd/vectalg/Matrix.hpp"
#include "softmax.h"
using namespace capd;

const interval AffineFunc::I = interval(-1,1);

interval AffineFunc::toInterval() const{
  auto i = coeffs.begin();
  interval result = i->second;
  for(++i;i!=coeffs.end();++i)
    result += i->second*I;
  return result;
}

void AffineFunc::addVar(const interval& c) { expr->addVar(*this,c); }

std::ostream& operator<<(std::ostream& out, const AffineFunc& f){
  int n = f.coeffs.rbegin()->first;
  auto b = f.coeffs.begin();
  out << "{" << b->second;
  while(++b!=f.coeffs.end())
    out << " (" << b->first << "," << b->second << ")";
  out << "}";
  return out;
}


AffineFunc operator+(const interval& c, const AffineFunc& f){
  AffineFunc result = f;
  result.coeffs[0] += c;
  return result;
}

AffineFunc operator+(const AffineFunc& f ,const interval& c){
  return c+f;
}

AffineFunc operator-(const interval& c, const AffineFunc& f){
  AffineFunc result(f.expr);
  auto i = f.coeffs.begin();
  result.coeffs[0] = c - i->second;
  for(++i;i!=f.coeffs.end();++i)
    result.coeffs[i->first] = -i->second;
  return result;
}

AffineFunc operator-(const AffineFunc& f, const interval& c){
  AffineFunc result = f;
  result.coeffs[0] -= c;
  return result;  
}

AffineFunc operator*(const interval& c, const AffineFunc& f){
  AffineFunc result(f.expr);
  for(auto i = f.coeffs.begin();i!=f.coeffs.end();++i)
    result.coeffs[i->first] = c*i->second;
  return result;
}

AffineFunc operator*(const AffineFunc& f ,const interval& c){
  return c*f;
}

AffineFunc operator+(const AffineFunc& f, const AffineFunc& g){
  AffineFunc result(AffineFunc::checkExpr(f,g));
  auto bf = f.coeffs.begin();
  auto bg = g.coeffs.begin();
  
  while(bf!=f.coeffs.end() and bg!=g.coeffs.end()){
    if(bf->first==bg->first){
      result.coeffs[bf->first] = bf->second + bg->second;
      ++bf;
      ++bg;
    } else if(bf->first < bg->first){
      result.coeffs[bf->first] = bf->second;
      ++bf;
    } else {
      result.coeffs[bf->first] = bg->second;
      ++bg;
    }
  }

  while(bf!=f.coeffs.end()){
    result.coeffs[bf->first] = bf->second;
    ++bf;
  }

  while(bg!=g.coeffs.end()){
    result.coeffs[bg->first] = bg->second;
    ++bg;
  }
  return result;
}

AffineFunc& AffineFunc::operator+=(const AffineFunc& g){
  this->expr = AffineFunc::checkExpr(*this,g);
  auto b = g.coeffs.begin();
  while(b!=g.coeffs.end()){
    this->coeffs[b->first] += b->second;
    ++b;
  }
  return *this;
}
    
AffineFunc operator-(const AffineFunc& f, const AffineFunc& g){
  AffineFunc result(AffineFunc::checkExpr(f,g));
  auto bf = f.coeffs.begin();
  auto bg = g.coeffs.begin();
  
  while(bf!=f.coeffs.end() and bg!=g.coeffs.end()){
    if(bf->first==bg->first){
      result.coeffs[bf->first] = bf->second - bg->second;
      ++bf;
      ++bg;
    } else if(bf->first < bg->first){
      result.coeffs[bf->first] = bf->second;
      ++bf;
    } else {
      result.coeffs[bf->first] = -bg->second;
      ++bg;
    }
  }

  while(bf!=f.coeffs.end()){
    result.coeffs[bf->first] = bf->second;
    ++bf;
  }

  while(bg!=g.coeffs.end()){
    result.coeffs[bg->first] = -bg->second;
    ++bg;
  }
  return result;
}

AffineFunc operator*(const AffineFunc& f, const AffineFunc& g){
  AffineFunc result(AffineFunc::checkExpr(f,g));
  
  auto bf = f.coeffs.begin();
  auto bg = g.coeffs.begin();

  if(result.expr==nullptr){
    result.coeffs[0] = bf->second*bg->second;
  } else {

    AffineFunc result = f*bg->second;
    interval sg = 0.0;
    while(++bg!=g.coeffs.end()){
      result.coeffs[bg->first] += bg->second*bf->second;
      sg += bg->second*AffineFunc::I;
    }
    
    interval sf = 0.0;
    while(++bf!=f.coeffs.end())
        sf += bf->second*AffineFunc::I;
        
    // Add a new symbol from quadratic terms
    // sf and sg are sums of all but contant terms in f and g, respectively
    result.addVar(sf*sg);
  }
  return result;
}

AffineFunc relu(const AffineFunc& f){
  interval c = f.toInterval();
  if(c<=0.0)
    return AffineFunc();
  if(c>=0.0)
    return f;

  // otherwise zero is in the interior of c
  auto i = f.coeffs.begin();
  if(f.expr==nullptr) return AffineFunc(interval(0.,i->second.rightBound()));

  interval e = c.right()/diam(c);
  // these two operation are just to make small 'e' closer to zero and larger 'e' (close to 1) much closer to one.
  e = 0.5*(1 - cos(capd::interval::pi()*e)).rightBound();
  e = 0.5*(1 - cos(capd::interval::pi()*e)).rightBound();

  interval a0 = i->second;
  interval S = 0.0;
  while(++i!=f.coeffs.end())
    S += abs(i->second).rightBound();
  interval M = a0+S;
  interval B = 0.5*e*M;
  c = B/S;
  interval D = abs(B-c*a0);
  
  AffineFunc result(f.expr);
  result.coeffs[0] = B;
  i = f.coeffs.begin();
  while(++i!=f.coeffs.end())
    result.coeffs[i->first] = c*i->second;
  result.addVar(intervalHull(-D,(1-e)*M));
  return result;

}

AffineVector relu(const AffineVector& u){
  AffineVector result(u.dimension(),false);
  for(int i=0;i<u.dimension();++i)
    result[i] = relu(u[i]);
  return result;
}

AffineVector softmax(AffineVector z){
  const int dim = z.dimension();
  IVector x(dim);
  // Prepare data. Store centre of affine set X in x0
  double R = 0.;
  for(unsigned i=0;i<dim;++i){
    x[i] = z[i].coeffs[0];
    R = capd::max(R,rightBound(x[i]));
  }
  // First compute softmax at the centre
  IVector s = softmax(x);
    
  // Compute bound on softmax using affine arithmetics reduction of coefficients
  IVector g(dim);
  for(int i=0;i<dim;++i){
    for(int j=0;j<dim;++j)
      g[i] += exp((z[j]-z[i]).toInterval()-R+x[i]);
    g[i] = exp(x[i]-R)/g[i];
  }
  
  // Try to improve this bound using the mean value form. We need to compute Dg.
  // First we compute off-diagonal part of the symmetric matrix.
  IMatrix Dg(dim,dim);
  for(int i=0;i<dim;++i){
    for(int c=i+1;c<dim;++c){
      for(int j=0;j<dim;++j)
        for(int k=0;k<dim;++k)
          Dg[i][c] += exp((z[j]+z[k]-z[i]-z[c]).toInterval()-2*R+x[i]+x[c]);
      Dg[c][i] = Dg[i][c] = -exp(x[i]+x[c]-2*R)/Dg[i][c];
    }
  }  
  // Diagonal elements are equal to g[i]*(1-g[i]) = 0.25-(g[i]-0.5)^2
  for(int i=0;i<dim;++i)
    Dg[i][i] = 0.25-sqr(g[i]-0.5);
  
  // Mean value form
  auto r = s + Dg*(z-x);
  // intersect with previous bound
  for(int i=0;i<dim;++i){
    interval ri = r[i].toInterval();
    if(!intersection(ri,g[i],g[i])) throw std::runtime_error("Error in softmax(AffineFunc): empty intersection");
    r[i] = AffineFunc(g[i]);
  }
  return r;
}