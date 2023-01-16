// Copyright (c) Tyler Veness

#include "MatrixUtils.hpp"

sleipnir::VariableMatrix expm(const sleipnir::VariableMatrix& mat) {
  assert(mat.Rows() == mat.Cols());

  sleipnir::VariableMatrix result{mat.Rows(), mat.Cols()};
  for (int row = 0; row < mat.Rows(); ++row) {
    result(row, row) = sleipnir::Variable{sleipnir::MakeConstant(1.0)};
  }

  sleipnir::VariableMatrix lastTerm{mat.Rows(), mat.Cols()};
  for (int row = 0; row < mat.Rows(); ++row) {
    lastTerm(row, row) = sleipnir::Variable{sleipnir::MakeConstant(1.0)};
  }
  for (int k = 1; k < 5; ++k) {
    lastTerm *= 1.0 / k * mat;
    result += lastTerm;
  }

  return result;
}
