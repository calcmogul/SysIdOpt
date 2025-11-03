// Copyright (c) Tyler Veness

#include "MatrixUtils.hpp"

slp::VariableMatrix<double> expm(const slp::VariableMatrix<double>& mat) {
  assert(mat.rows() == mat.cols());

  slp::VariableMatrix<double> result{mat.rows(), mat.cols()};
  for (int row = 0; row < mat.rows(); ++row) {
    result[row, row] = 1.0;
  }

  slp::VariableMatrix<double> lastTerm{mat.rows(), mat.cols()};
  for (int row = 0; row < mat.rows(); ++row) {
    lastTerm[row, row] = 1.0;
  }
  for (int k = 1; k < 5; ++k) {
    lastTerm *= 1.0 / k * mat;
    result += lastTerm;
  }

  return result;
}
