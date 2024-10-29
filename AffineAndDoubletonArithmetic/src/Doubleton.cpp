/////////////////////////////////////////////////////////////////////////////
/// @file Doubleton.cpp
///
/// @author (C) 2024 Daniel Wilczak
///
/// This file is distributed under the terms of the GNU General Public License.

/// This file contains implementation of doubleton arithmetics.

#include "Doubleton.h"
#include "softmax.h"
using namespace capd;

IMatrix findQ(const IMatrix& B, const IVector& q, const IVector& z){
  const int d = z.dimension();
  IMatrix Q(d,d);
  Q.setToIdentity();
  return Q;
  
  std::vector< std::pair<double,int> > perm(d);
  for(int i=0;i<d;++i)
    perm[i] = i < q.dimension()? 
      std::make_pair( (width(z[i]) + B.column(i).euclNorm()*width(q[i]) ).rightBound(),i) : 
      std::make_pair(width(z[i]),i);
  std::sort(perm.rbegin(),perm.rend());
  
  IMatrix M = midMatrix(B);
  for(int i=0;i<d;++i){
    IVector column = perm[i].second < M.numberOfColumns() ? M.column(perm[i].second) : Q.column(i);
    
    for(int j=0;j<i;++j){
      interval p = column*Q.column(j);
      Q.column(i) -= Q.column(j)*p;
    }
    interval n = Q.column(i).euclNorm();
    if(!isSingular(n)){
      if(n.leftBound()>n.rightBound()){
        std::cout << Q.column(i);
        exit(0);
      }
      Q.column(i)/=n;
      ++i;
      continue;
    } else {
      throw "singular";
      M.column(perm[i].second)[i]+= 1;
      Q.column(i) = M.column(perm[i].second);
    }
  }
  return Q;
}
  
void absorb(const IVector& delta, IMatrix& Q, IVector& q){
  IVector z = Q*q+delta;
  IMatrix B = Q;
  Q = findQ(B,q,delta);
  IMatrix QT = Transpose(Q);
  q = (QT*B)*q + QT*delta;    
  intersection(q,QT*z,q);
}

Doubleton relu(Doubleton d){
  IVector u = d.toVector();
  const int dim = u.dimension();
  IVector y(dim);

  bool foundZero = false;
  for(int i=0;i<dim;++i) {
    if(u[i]<=0.0){
      d.x[i] = 0.0;
      d.C.row(i).clear();
      d.Q.row(i).clear();
      continue;
    }
    if(u[i]>=0.0) continue;
    foundZero = true;

    interval S = 0.0;
    for(int j=0;j<d.C.numberOfColumns();++j)
      S += abs(d.C[i][j]*d.r[j]).rightBound();
    for(int j=0;j<d.Q.numberOfColumns();++j)
      S += abs(d.Q[i][j]*d.q[j]).rightBound();
    interval M = d.x[i]+S;
    interval B = 0.5*M;
    interval c = B/S;
    interval D = 0.5*abs(B-c*d.x[i]);
    
    d.x[i] = B - D;
    d.C.row(i) *=c;
    d.Q.row(i) *=c;
    y[i] = interval(-1,1)*D;
  }
  if(foundZero) absorb(y,d.Q,d.q);
  return d;
}

Doubleton softmax(const Doubleton& d){
  const int dim = d.x.dimension();
  IVector s = softmax(d.x);
    
  double R = 0.;
  for(unsigned i=0;i<dim;++i){
    R = capd::max(R,rightBound(d.x[i]));
  }
  IVector g(dim);
  for(int i=0;i<dim;++i){
    for(int j=0;j<dim;++j){
      interval r = (d.C.row(j)-d.C.row(i))*d.r;
      interval q = (d.Q.row(j)-d.Q.row(i))*d.q;
      g[i] += exp(d.x[j]-d.x[i] + r + q -2*R);
    }
    g[i] = exp(-2*R)/g[i];
  }

  IMatrix Dg(dim,dim);
  for(int i=0;i<dim;++i){
    for(int c=i+1;c<dim;++c){
      for(int j=0;j<dim;++j)
        for(int k=0;k<dim;++k){
          interval r = (d.C.row(j)+d.C.row(k)-d.C.row(i)-d.C.row(c))*d.r;
          interval q = (d.Q.row(j)+d.Q.row(k)-d.Q.row(i)-d.Q.row(c))*d.q;
          Dg[i][c] += exp(d.x[j]+d.x[k]-d.x[i]-d.x[c] + r + q -4*R);
        }
      Dg[c][i] = Dg[i][c] = -exp(-4*R)/Dg[i][c];
    }
  }  
  // Diagonal elements are equal to g[i]*(1-g[i]) = 0.25-(g[i]-0.5)^2
  for(int i=0;i<dim;++i)
    Dg[i][i] = 0.25-sqr(g[i]-0.5);

  // Mean value form
  IVector z = s + (Dg*d.C)*d.r + (Dg*d.Q)*d.q;
  // intersect with previous bound
  for(int i=0;i<dim;++i){
    if(!intersection(z[i],g[i],g[i])) {
      throw std::runtime_error("Error in softmax(Doubleton): empty intersection");
    }
  }
  return Doubleton(g);
}