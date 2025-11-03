// Copyright (c) Tyler Veness

#pragma once

#include <Eigen/Core>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <units/time.h>
#include <unsupported/Eigen/MatrixFunctions>

/**
 * Undiscretizes the given continuous A and B matrices.
 *
 * @tparam States Number of states.
 * @tparam Inputs Number of inputs.
 * @param discA Discrete system matrix.
 * @param discB Discrete input matrix.
 * @param dt    Discretization timestep.
 * @param contA Storage for continuous system matrix.
 * @param contB Storage for continuous input matrix.
 */
template <int States, int Inputs>
void UndiscretizeAB(const Eigen::Matrix<double, States, States>& discA,
                    const Eigen::Matrix<double, States, Inputs>& discB,
                    units::second_t dt,
                    Eigen::Matrix<double, States, States>* contA,
                    Eigen::Matrix<double, States, Inputs>* contB) {
  // ϕ = [A_d  B_d]
  //     [ 0    I ]
  Eigen::Matrix<double, States + Inputs, States + Inputs> phi;
  phi.template block<States, States>(0, 0) = discA;
  phi.template block<States, Inputs>(0, States) = discB;
  phi.template block<Inputs, States>(States, 0).setZero();
  phi.template block<Inputs, Inputs>(States, States).setIdentity();

  // M = log(ϕ/T) = [A  B]  // NOLINT
  //                [0  0]
  decltype(phi) M = phi.log() / dt.value();

  *contA = M.template block<States, States>(0, 0);
  *contB = M.template block<States, Inputs>(0, States);
}

/**
 * Performs the matrix exponential of a matrix.
 *
 * @param mat The matrix to exponentiate.
 */
slp::VariableMatrix<double> expm(const slp::VariableMatrix<double>& mat);
