// Copyright (c) Tyler Veness

#pragma once

#include <Eigen/Core>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <wpi/units/time.hpp>

/// Undiscretizes the given continuous A and B matrices.
///
/// @tparam States Number of states.
/// @tparam Inputs Number of inputs.
/// @param disc_A Discrete system matrix.
/// @param disc_B Discrete input matrix.
/// @param dt Discretization timestep.
/// @param cont_A Storage for continuous system matrix.
/// @param cont_B Storage for continuous input matrix.
template <int States, int Inputs>
void undiscretize_ab(const Eigen::Matrix<double, States, States>& disc_A,
                     const Eigen::Matrix<double, States, Inputs>& disc_B,
                     wpi::units::second_t dt,
                     Eigen::Matrix<double, States, States>* cont_A,
                     Eigen::Matrix<double, States, Inputs>* cont_B) {
  // ϕ = [A_d  B_d]
  //     [ 0    I ]
  Eigen::Matrix<double, States + Inputs, States + Inputs> phi;
  phi.template block<States, States>(0, 0) = disc_A;
  phi.template block<States, Inputs>(0, States) = disc_B;
  phi.template block<Inputs, States>(States, 0).setZero();
  phi.template block<Inputs, Inputs>(States, States).setIdentity();

  // M = log(ϕ/T) = [A  B]
  //                [0  0]
  decltype(phi) M = phi.log() / dt.value();

  *cont_A = M.template block<States, States>(0, 0);
  *cont_B = M.template block<States, Inputs>(0, States);
}

/// Performs the matrix exponential of a matrix.
///
/// @param mat The matrix to exponentiate.
slp::VariableMatrix<double> expm(const slp::VariableMatrix<double>& mat);
