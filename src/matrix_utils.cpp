// Copyright (c) Tyler Veness

#include "matrix_utils.hpp"

slp::VariableMatrix<double> expm(const slp::VariableMatrix<double>& mat) {
  assert(mat.rows() == mat.cols());

  slp::VariableMatrix<double> result{mat.rows(), mat.cols()};
  for (int row = 0; row < mat.rows(); ++row) {
    result[row, row] = 1.0;
  }

  slp::VariableMatrix<double> last_term{mat.rows(), mat.cols()};
  for (int row = 0; row < mat.rows(); ++row) {
    last_term[row, row] = 1.0;
  }
  for (int k = 1; k < 5; ++k) {
    last_term *= 1.0 / k * mat;
    result += last_term;
  }

  return result;
}
