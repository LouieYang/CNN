#ifndef ANN_LoadMatrix_h
#define ANN_LoadMatrix_h

#include <eigen3/Eigen/Eigen>
#include <string>
#include <fstream>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::VectorXf;

void LoadMatrix(Eigen::MatrixXf& matrix, std::string address,
                const int cols, const int rows);
#endif
