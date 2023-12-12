// Copyright (c) Tyler Veness

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <fmt/core.h>
#include <frc/fmt/Eigen.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/length.h>
#include <units/time.h>
#include <units/velocity.h>
#include <wpi/MemoryBuffer.h>
#include <wpi/json.h>

#include "MatrixUtils.hpp"

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
double ToMilliseconds(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1000.0;
}

struct FeedforwardGains {
  double Ks = 0.0;
  double Kv = 0.0;
  double Ka = 0.0;
};

/**
 * Solves SysId's OLS problem with Eigen to produce initial guess for nonlinear
 * problem.
 *
 * @param[in] json SysId JSON.
 * @param[in] motionThreshold Data with velocities closer to zero than this are
 *   ignored.
 * @return Initial guess for nonlinear problem.
 */
FeedforwardGains SolveEigenSysIdOLS(
    const wpi::json& json, units::meters_per_second_t motionThreshold) {
  // Find average timestep
  double T = 0.0;
  int samples = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      T += t_k1 - t_k;
      ++samples;
    }
  }
  T /= static_cast<double>(samples);

  Eigen::MatrixXd X{samples, 3};
  Eigen::MatrixXd y{samples, 1};

  int sample = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    // See
    // https://github.com/wpilibsuite/sysid/blob/main/docs/data-collection.md
    //
    // Non-Drivetrain Mechanisms:
    //   timestamp, voltage, position, velocity
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, x_k] = data[k];
      auto& [t_k1, u_k1, p_k1, x_k1] = data[k + 1];

      // Ignore data with velocity below motion threshold
      if (std::abs(x_k) < motionThreshold.value()) {
        continue;
      }

      // Add the velocity term (for alpha)
      X(sample, 0) = x_k;

      // Add the voltage term (for beta)
      X(sample, 1) = u_k;

      // Add the intercept term (for gamma)
      X(sample, 2) = std::copysign(1.0, x_k);

      // Add the dependent variable (acceleration)
      y(sample, 0) = (x_k1 - x_k) / T;

      ++sample;
    }
  }

  // Solve (XᵀX)β = Xᵀy
  Eigen::MatrixXd b = (X.transpose() * X).llt().solve(X.transpose() * y);

  double alpha = b(0, 0);
  double beta = b(1, 0);
  double gamma = b(2, 0);

  return {-gamma / beta, -alpha / beta, 1.0 / beta};
}

/**
 * Solves linear system ID problem with Eigen to produce initial guess for
 * nonlinear problem.
 *
 * @param[in] json SysId JSON.
 * @param[in] motionThreshold Data with velocities closer to zero than this are
 *   ignored.
 * @param[in] positionStddev Position standard deviation.
 * @param[in] velocityStddev Velocity standard deviation.
 * @return Initial guess for nonlinear problem.
 */
FeedforwardGains SolveEigenLinearSystem(
    const wpi::json& json, units::meters_per_second_t motionThreshold,
    units::meter_t positionStddev, units::meters_per_second_t velocityStddev) {
  // [p]    = [1  a][p]  + [0]   + [0]
  // [v]ₖ₊₁   [0  b][v]ₖ   [c]uₖ   [d]sgn(vₖ)
  //
  // [p]    = [pₖ] + [vₖa] + [ 0 ] + [   0    ]
  // [v]ₖ₊₁   [0 ]   [vₖb]   [uₖc]   [sgn(vₖ)d]
  //
  // [pₖ₊₁ - pₖ] = [vₖa] + [ 0 ] + [   0    ]
  // [  vₖ₊₁   ]   [vₖb]   [uₖc]   [sgn(vₖ)d]
  //
  // [pₖ₊₁ - pₖ] = [vₖa] + [ 0 ] + [ 0 ] + [   0    ]
  // [  vₖ₊₁   ]   [ 0 ]   [vₖb]   [uₖc]   [sgn(vₖ)d]
  //
  // [pₖ₊₁ - pₖ] = [vₖ]  + [0 ]  + [0 ]  + [   0   ]
  // [  vₖ₊₁   ]   [0 ]a   [vₖ]b   [uₖ]c   [sgn(vₖ)]d
  //
  //                                   [a]
  // [pₖ₊₁ - pₖ] = [vₖ  0   0     0   ][b]
  // [  vₖ₊₁   ]   [0   vₖ  uₖ sgn(vₖ)][c]
  //                                   [d]
  //
  //     [vₖ  0   0     0   ]
  // X = [0   vₖ  uₖ sgn(vₖ)]
  //     [        ⋮         ]
  //
  //     [pₖ₊₁ - pₖ]
  // y = [  vₖ₊₁   ]
  //     [    ⋮    ]
  //
  // W = diag([1/σₚ², 1/σᵥ², …])
  //
  //     [a]
  // β = [b]
  //     [c]
  //     [d]
  //
  // argmin rᵀWr
  //   β
  // where r = y - Xβ
  //
  // d/dβ (y - Xβ)ᵀW(y - Xβ) = 0
  // -2XᵀW(y - Xβ) = 0
  // XᵀW(y - Xβ) = 0
  // XᵀWy - XᵀWXβ = 0
  // XᵀWXβ = XᵀWy
  //
  // Let N be the number of samples.
  //
  // dim(y) = 2N x 1
  // dim(X) = 2N x 4
  // dim(β) = 4 x 1
  // dim(W) = 2N x 2N

  constexpr int States = 2;
  constexpr int Inputs = 1;

  // Find average timestep
  double T = 0.0;
  int samples = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      T += t_k1 - t_k;
      ++samples;
    }
  }
  T /= static_cast<double>(samples);

  Eigen::MatrixXd X{2 * samples, 4};
  Eigen::MatrixXd y{2 * samples, 1};

  int sample = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    // See
    // https://github.com/wpilibsuite/sysid/blob/main/docs/data-collection.md
    //
    // Non-Drivetrain Mechanisms:
    //   timestamp, voltage, position, velocity
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      // Ignore data with velocity below motion threshold
      if (std::abs(v_k) < motionThreshold.value()) {
        continue;
      }

      //     [vₖ  0   0     0   ]
      // X = [0   vₖ  uₖ sgn(vₖ)]
      //     [        ⋮         ]
      X.block(2 * sample, 0, 2, 4) = Eigen::Matrix<double, 2, 4>{
          {v_k, 0.0, 0.0, 0.0}, {0.0, v_k, u_k, std::copysign(1.0, v_k)}};

      //     [pₖ₊₁ - pₖ]
      // y = [  vₖ₊₁   ]
      //     [    ⋮    ]
      y(2 * sample, 0) = p_k1 - p_k;
      y(2 * sample + 1, 0) = v_k1;

      ++sample;
    }
  }

  // XᵀW where W = diag([1/σₚ², 1/σᵥ², …])
  Eigen::MatrixXd Xweighted = X.transpose();
  for (int col = 0; col < Xweighted.cols(); ++col) {
    Xweighted.col(col) *= 1.0 / std::pow(positionStddev.value(), 2);
    Xweighted.col(col + 1) *= 1.0 / std::pow(velocityStddev.value(), 2);
  }

  // Solve (XᵀWX)β = XᵀWy
  Eigen::MatrixXd beta = (Xweighted * X).llt().solve(Xweighted * y);

  double a = beta(0, 0);
  double b = beta(1, 0);
  double c = beta(2, 0);
  double d = beta(3, 0);

  Eigen::Matrix<double, States, States> discA{{1.0, a}, {0.0, b}};
  Eigen::Matrix<double, States, Inputs> discB{{0.0}, {c}};
  Eigen::Matrix<double, States, Inputs> discC{{0.0}, {d}};
  Eigen::Matrix<double, States, States> contA;
  Eigen::Matrix<double, States, Inputs> contB;
  Eigen::Matrix<double, States, Inputs> contC;
  UndiscretizeAB(discA, discB, units::second_t{T}, &contA, &contB);
  UndiscretizeAB(discA, discC, units::second_t{T}, &contA, &contC);

  //              A           B          c
  //         [0     1  ]    [ 0  ]    [  0   ]
  // dx/dt = [0  -Kv/Ka]x + [1/Ka]u + [-Ks/Ka]sgn(x)

  // Ks, Kv, Ka
  return {-contC(1, 0) / contB(1, 0), -contA(1, 1) / contB(1, 0),
          1.0 / contB(1, 0)};
}

/**
 * Solves SysId's OLS problem with Sleipnir to produce initial guess for
 * nonlinear problem.
 *
 * @param[in] json SysId JSON.
 * @param[in] motionThreshold Data with velocities closer to zero than this are
 *   ignored.
 * @return Initial guess for nonlinear problem.
 */
FeedforwardGains SolveSleipnirSysIdOLS(
    const wpi::json& json, units::meters_per_second_t motionThreshold) {
  // Implements https://file.tavsys.net/control/sysid-ols.pdf

  // Find average timestep
  double T = 0.0;
  int samples = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      T += t_k1 - t_k;
      ++samples;
    }
  }
  T /= static_cast<double>(samples);

  sleipnir::OptimizationProblem problem;

  auto alpha = problem.DecisionVariable();
  auto beta = problem.DecisionVariable();
  auto gamma = problem.DecisionVariable();

  sleipnir::Variable J = 0.0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    // See
    // https://github.com/wpilibsuite/sysid/blob/main/docs/data-collection.md
    //
    // Non-Drivetrain Mechanisms:
    //   timestamp, voltage, position, velocity
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, x_k] = data[k];
      auto& [t_k1, u_k1, p_k1, x_k1] = data[k + 1];

      // Ignore data with velocity below motion threshold
      if (std::abs(x_k) < motionThreshold.value()) {
        continue;
      }

      auto f = [&](const auto& x, const auto& u) {
        return alpha * x + beta * u + gamma * std::copysign(1.0, x);
      };

      J += sleipnir::pow(x_k1 - f(x_k, u_k), 2);
    }
  }
  problem.Minimize(J);

  problem.SubjectTo(alpha > 0);

  problem.Solve();

  return {-gamma.Value() / beta.Value(), (1.0 - alpha.Value()) / beta.Value(),
          (alpha.Value() - 1.0) * T / (beta.Value() * std::log(alpha.Value()))};
}

/**
 * Solves linear system ID problem with Sleipnir to produce initial guess for
 * nonlinear problem.
 *
 * @param[in] json SysId JSON.
 * @param[in] motionThreshold Data with velocities closer to zero than this are
 *   ignored.
 * @param[in] positionStddev Position standard deviation.
 * @param[in] velocityStddev Velocity standard deviation.
 * @return Initial guess for nonlinear problem.
 */
FeedforwardGains SolveSleipnirLinearSystem(
    const wpi::json& json, units::meters_per_second_t motionThreshold,
    units::meter_t positionStddev, units::meters_per_second_t velocityStddev) {
  constexpr int States = 2;
  constexpr int Inputs = 1;

  // Find average timestep
  double T = 0.0;
  int samples = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      T += t_k1 - t_k;
      ++samples;
    }
  }
  T /= static_cast<double>(samples);

  sleipnir::OptimizationProblem problem;

  auto a = problem.DecisionVariable();
  auto b = problem.DecisionVariable();
  auto c = problem.DecisionVariable();
  auto d = problem.DecisionVariable();

  sleipnir::Variable J = 0.0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    // See
    // https://github.com/wpilibsuite/sysid/blob/main/docs/data-collection.md
    //
    // Non-Drivetrain Mechanisms:
    //   timestamp, voltage, position, velocity
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      // Ignore data with velocity below motion threshold
      if (std::abs(v_k) < motionThreshold.value()) {
        continue;
      }

      // See equation (2.11) of
      // https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.pdf
      double pWeight = 1.0 / std::pow(positionStddev.value(), 2);
      double vWeight = 1.0 / std::pow(velocityStddev.value(), 2);

      //          A         B       c
      //        [1  a]     [0]     [0]
      // xₖ₊₁ = [0  b]xₖ + [c]uₖ + [d]sgn(xₖ)
      J += pWeight * sleipnir::pow(p_k1 - (p_k + a * v_k), 2);
      J += vWeight *
           sleipnir::pow(
               v_k1 - (b * v_k + c * u_k + d * std::copysign(1.0, v_k)), 2);
    }
  }
  problem.Minimize(J);

  problem.Solve();

  Eigen::Matrix<double, States, States> discA{{1.0, a.Value()},
                                              {0.0, b.Value()}};
  Eigen::Matrix<double, States, Inputs> discB{{0.0}, {c.Value()}};
  Eigen::Matrix<double, States, Inputs> discC{{0.0}, {d.Value()}};
  Eigen::Matrix<double, States, States> contA;
  Eigen::Matrix<double, States, Inputs> contB;
  Eigen::Matrix<double, States, Inputs> contC;
  UndiscretizeAB(discA, discB, units::second_t{T}, &contA, &contB);
  UndiscretizeAB(discA, discC, units::second_t{T}, &contA, &contC);

  //              A           B          c
  //         [0     1  ]    [ 0  ]    [  0   ]
  // dx/dt = [0  -Kv/Ka]x + [1/Ka]u + [-Ks/Ka]sgn(x)

  // Ks, Kv, Ka
  return {-contC(1, 0) / contB(1, 0), -contA(1, 1) / contB(1, 0),
          1.0 / contB(1, 0)};
}

/**
 * Solves nonlinear system ID problem with Sleipnir.
 *
 * @param[in] json SysId JSON.
 * @param[in] motionThreshold Data with velocities closer to zero than this are
 *   ignored.
 * @param[in] positionStddev Position standard deviation.
 * @param[in] velocityStddev Velocity standard deviation.
 * @param[in] initialGuess Initial guess from linear problem.
 */
FeedforwardGains SolveSleipnirNonlinear(
    const wpi::json& json, units::meters_per_second_t motionThreshold,
    units::meter_t positionStddev, units::meters_per_second_t velocityStddev,
    const FeedforwardGains& initialGuess) {
  constexpr int States = 2;
  constexpr int Inputs = 1;

  sleipnir::OptimizationProblem problem;

  auto Ks = problem.DecisionVariable();
  Ks = initialGuess.Ks;

  auto Kv = problem.DecisionVariable();
  Kv = initialGuess.Kv;

  auto Ka = problem.DecisionVariable();
  Ka = initialGuess.Ka;

  sleipnir::Variable J = 0.0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    // See
    // https://github.com/wpilibsuite/sysid/blob/main/docs/data-collection.md
    //
    // Non-Drivetrain Mechanisms:
    //   timestamp, voltage, position, velocity
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      // Ignore data with velocity below motion threshold
      if (std::abs(v_k) < motionThreshold.value()) {
        continue;
      }

      Eigen::Vector<double, States> x{{p_k}, {v_k}};
      Eigen::Vector<double, States> x_next{{p_k1}, {v_k1}};
      Eigen::Vector<double, Inputs> u{u_k};

      double T = t_k1 - t_k;

      // dx/dt = Ax + Bu + c
      // xₖ₊₁ = eᴬᵀxₖ + A⁻¹(eᴬᵀ − 1)(Buₖ + c)
      // xₖ₊₁ = A_d xₖ + A⁻¹(A_d − 1)(Buₖ + c)
      sleipnir::VariableMatrix A{{0, 1}, {0, -Kv / Ka}};
      sleipnir::VariableMatrix B{{0}, {1 / Ka}};
      sleipnir::VariableMatrix c{{0}, {-Ks / Ka}};

      // Discretize model without B so it can be reused for c
      sleipnir::VariableMatrix M{States + Inputs, States + Inputs};
      M.Block(0, 0, States, States) = A;
      for (int row = 0; row < std::min(States, Inputs); ++row) {
        M(row, States + row) = 1.0;
      }
      auto phi = expm(M * T);

      auto A_d = phi.Block(0, 0, States, States);
      auto B_d = phi.Block(0, States, States, Inputs) * B;
      auto c_d = phi.Block(0, States, States, Inputs) * c;
      auto f = [&](const auto& x, const auto& u) {
        return A_d * x + B_d * u + c_d * std::copysign(1.0, x(1, 0));
      };

      Eigen::Matrix<double, 2, 2> sigmaInv{
          {1.0 / std::pow(positionStddev.value(), 2), 0.0},
          {0.0, 1.0 / std::pow(velocityStddev.value(), 2)}};

      J += (x_next - f(x, u)).T() * sigmaInv * (x_next - f(x, u));
    }
  }
  problem.Minimize(J);

  problem.Solve();

  return {Ks.Value(), Kv.Value(), Ka.Value()};
}

/**
 * Runs the given solver.
 *
 * @param name Name to print for results.
 * @param solver Solver that returns feedforward gains.
 */
FeedforwardGains RunSolve(std::string_view name,
                          std::function<FeedforwardGains()> solver) {
  fmt::print("{}\n", name);

  auto startTime = std::chrono::system_clock::now();
  FeedforwardGains gains = solver();
  auto endTime = std::chrono::system_clock::now();

  fmt::print("  duration = {} ms\n", ToMilliseconds(endTime - startTime));
  fmt::print("  Ks = {}\n", gains.Ks);
  fmt::print("  Kv = {}\n", gains.Kv);
  fmt::print("  Ka = {}\n", gains.Ka);

  return gains;
}

int main(int argc, const char* argv[]) {
  constexpr units::meters_per_second_t kMotionThreshold = 0.1_mps;
  constexpr units::meter_t kPositionStddev = 1_cm;
  constexpr units::meters_per_second_t kVelocityStddev = 0.2_mps;

  std::span args(argv, argc);

  if (args.size() == 1) {
    fmt::print(stderr, "Specify a JSON filename.\n");
    return 1;
  }

  // Read JSON from the specified path
  wpi::json json;
  {
    std::error_code ec;
    auto fileBuffer = wpi::MemoryBuffer::GetFile(args[1], ec);
    if (fileBuffer == nullptr || ec) {
      fmt::print(stderr, "Failed to open file '{}'\n", args[1]);
      return 1;
    }

    json = wpi::json::parse(fileBuffer->GetCharBuffer());
  }

  RunSolve("Eigen SysId OLS (velocity only)",
           [&] { return SolveEigenSysIdOLS(json, kMotionThreshold); });
  RunSolve("Eigen LinearSystem (position and velocity)", [&] {
    return SolveEigenLinearSystem(json, kMotionThreshold, kPositionStddev,
                                  kVelocityStddev);
  });
  RunSolve("Sleipnir SysId OLS (velocity only)",
           [&] { return SolveSleipnirSysIdOLS(json, kMotionThreshold); });
  auto initialGuess =
      RunSolve("Sleipnir LinearSystem (position and velocity)", [&] {
        return SolveSleipnirLinearSystem(json, kMotionThreshold,
                                         kPositionStddev, kVelocityStddev);
      });
  RunSolve("Sleipnir nonlinear (position and velocity)", [&] {
    return SolveSleipnirNonlinear(json, kMotionThreshold, kPositionStddev,
                                  kVelocityStddev, initialGuess);
  });
}
