/////////////////////////////////////////////////////////////////////////////
/// @file ParseNN.cpp

/// @author (C) 2024 Daniel Wilczak
///
/// This file is distributed under the terms of the GNU General Public License.

/// This file contains implementation of auxiliary routines for parsing a neural network 
/// from text files in various formats, including json.

#include <fstream>
#include <sstream>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include "NeuralNetwork.h"

using namespace std;
using namespace capd;
using namespace rapidjson;

DVector parseBias(Document& d, const char* key){
  const auto& a = d[key];
  DVector result(a.Size());
  for(unsigned i=0;i<result.dimension();++i){
    result[i] = a[i].GetDouble();
  }
  return result;
}

DMatrix parseMatrix(Document& d, const char* key){
  const auto& a = d[key];
  DMatrix result(a.Size(),a[0].Size());
  for(unsigned i=0;i<result.numberOfRows();++i)
    for(unsigned j=0;j<result.numberOfColumns();++j)
      result[i][j] = a[i][j].GetDouble();
  return result;
}

DTensor parseTensor(Document& d, const char* key){
  const auto& a = d[key];
  DTensor result;
  for(int k=0;k<a.Size();++k){
    std::vector<DMatrix> r;
    for(int s=0;s<a[k].Size();++s){
      DMatrix m(a[k][s].Size(),a[k][s][0].Size());
      for(unsigned i=0;i<m.numberOfRows();++i)
        for(unsigned j=0;j<m.numberOfColumns();++j)
          m[i][j] = a[k][s][i][j].GetDouble();
      r.push_back(m);
    }
    result.push_back(r);
  }
  return result;
}

NeuralNetworkLayer* parseLayer(Document& d, const char* wKey, const char* bKey){
  const auto& a = d[wKey];
  NeuralNetworkLayer* result;
  if(a[0][0].IsNumber()){
    IMatrix A = capd::vectalg::convertObject<IMatrix>(parseMatrix(d,wKey));
    IVector b = capd::vectalg::convertObject<IVector>(parseBias(d,bKey));
    result =  new AffineLayer(A,b); 
  } else {
    result = new ConvolutionalLayer(parseTensor(d,wKey),parseBias(d,bKey));
  }
  return result;
}

void parseFromFile(NeuralNetwork& nn, const char* filename, bool addSoftMaxLayer){
  Document document;
  ifstream in(filename);
  string s;
  getline(in,s);
  in.close();
  document.Parse(s.c_str());

  vector<NeuralNetworkLayer*> layers;
  for (auto i = document.MemberBegin(); i != document.MemberEnd();++i)
  {
    std::string w = i->name.GetString();
    ++i;
    std::string b = i->name.GetString();
    layers.push_back(parseLayer(document,w.c_str(),b.c_str()));
  }
  nn.add(layers[0]);
  for(unsigned i=1;i<layers.size();++i){
    nn.add(new ReluLayer());
    nn.add(layers[i]);
  }
  
  if(addSoftMaxLayer)
    nn.add(new SoftMaxLayer());
}

list<DVector> readFullyConnectedDataSet(const char* filename) {
    // Result list of DVectors
    list<DVector> result;

    // Open file stream and use RapidJSON to parse
    ifstream ifs(filename);
    if (!ifs.is_open()) {
        cerr << "Could not open file: " << filename << endl;
        return result;
    }
    
    // Wrap the file stream into an input stream for RapidJSON
    IStreamWrapper isw(ifs);
    Document doc;
    doc.ParseStream(isw);

    // Check if the JSON is an object
    if (!doc.IsObject()) {
        cerr << "JSON parsing error: Root is not an object" << endl;
        return result;
    }

    // Iterate through each key-value pair in the JSON
    for (auto& member : doc.GetObject()) {
        const Value& array = member.value;

        // Ensure each value is an array of arrays
        if (array.IsArray() && array[0].IsArray()) {
            for (const auto& innerArray : array.GetArray()) {
                // Get the dimension of the vector
                size_t dim = innerArray.Size();

                // Create a DVector and fill it with values from the JSON
                DVector u(dim);
                for (size_t i = 0; i < dim; ++i) {
                    u[i] = innerArray[i].GetDouble();  // Assuming DVector supports this
                }

                // Add the vector to the result list
                result.push_back(u);
            }
        }
    }

    ifs.close();
    return result;
}

list<DTensor> readConvolutionalDataSet(const char* filename){
  Document document;
  ifstream in(filename);
  string s;
  getline(in,s);
  in.close();
  document.Parse(s.c_str());

  list<DTensor> result;
  for (auto i = document.MemberBegin(); i != document.MemberEnd();++i)
  {
    std::string w = i->name.GetString();
    if(i->value[0][0].IsArray())
      result.push_back(parseTensor(document,i->name.GetString()));
  }
  return result;
  
}
