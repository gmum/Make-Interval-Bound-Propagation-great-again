#include <list>
#include <rapidjson/document.h>
#include "NeuralNetwork.h"

#ifndef __CAPD_PARSE_NN__
#define __CAPD_PARSE_NN__

void parseFromFile(NeuralNetwork& nn, const char* filename, bool addSoftMaxLayer);

capd::DVector parseBias(rapidjson::Document& d, const char* key);
capd::DMatrix parseMatrix(rapidjson::Document& d, const char* key);
DTensor parseTensor(rapidjson::Document& d, const char* key);

std::list<capd::DVector> readDataSet(const char* filename, int dim);
std::list<DTensor> readDataSet(const char* filename);

#endif