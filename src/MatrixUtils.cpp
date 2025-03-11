// Copyright (c) Tyler Veness

#include "MatrixUtils.hpp"

slp::VariableMatrix expm(const slp::VariableMatrix& mat) {
  assert(mat.rows() == mat.cols());

  slp::VariableMatrix result{mat.rows(), mat.cols()};
  for (int row = 0; row < mat.rows(); ++row) {
    result[row, row] = 1.0;
  }

  slp::VariableMatrix lastTerm{mat.rows(), mat.cols()};
  for (int row = 0; row < mat.rows(); ++row) {
    lastTerm[row, row] = 1.0;
  }
  for (int k = 1; k < 5; ++k) {
    lastTerm *= 1.0 / k * mat;
    result += lastTerm;
  }

  return result;
}
