/////////////////////////////////////////////////////////////////////////////
/// @file ParseNN.h
///
/// @author (C) 2024 Daniel Wilczak
///
/// This file is distributed under the terms of the GNU General Public License.

#include <list>
#include <rapidjson/document.h>
#include "NeuralNetwork.h"

#ifndef __CAPD_PARSE_NN__
#define __CAPD_PARSE_NN__

void parseFromFile(NeuralNetwork& nn, const char* filename, bool addSoftMaxLayer);

capd::DVector parseBias(rapidjson::Document& d, const char* key);
capd::DMatrix parseMatrix(rapidjson::Document& d, const char* key);
DTensor parseTensor(rapidjson::Document& d, const char* key);

std::list<capd::DVector> readFullyConnectedDataSet(const char* filename);
std::list<DTensor> readConvolutionalDataSet(const char* filename);

#endif