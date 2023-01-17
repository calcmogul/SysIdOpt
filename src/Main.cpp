// Copyright (c) Tyler Veness

#include <array>
#include <chrono>
#include <cmath>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <fmt/core.h>
#include <frc/fmt/Eigen.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/length.h>
#include <units/time.h>
#include <units/velocity.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <wpi/json.h>
#include <wpi/raw_istream.h>

#include "MatrixUtils.hpp"

double sign(double x) {
  if (x > 0.0) {
    return 1.0;
  } else if (x < 0.0) {
    return -1.0;
  } else {
    return 0.0;
  }
}

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
  int elems = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      T += t_k1 - t_k;
      ++elems;
    }
  }
  T /= static_cast<double>(elems);

  Eigen::MatrixXd X{elems, 3};
  Eigen::MatrixXd y{elems, 1};

  int elem = 0;
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
      X(elem, 0) = x_k;

      // Add the voltage term (for beta)
      X(elem, 1) = u_k;

      // Add the intercept term (for gamma)
      X(elem, 2) = std::copysign(1.0, x_k);

      // Add the dependent variable (acceleration)
      y(elem, 0) = (x_k1 - x_k) / T;

      ++elem;
    }
  }

  Eigen::MatrixXd b = (X.transpose() * X).llt().solve(X.transpose() * y);

  double alpha = b(0, 0);
  double beta = b(1, 0);
  double gamma = b(2, 0);

  return {-gamma / beta, -alpha / beta, 1.0 / beta};
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
  int elems = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      T += t_k1 - t_k;
      ++elems;
    }
  }
  T /= static_cast<double>(elems);

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
        return alpha * x + beta * u + gamma * sign(x);
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
  int elems = 0;
  for (auto&& testName :
       {"fast-backward", "fast-forward", "slow-backward", "slow-forward"}) {
    auto data = json.at(testName).get<std::vector<std::array<double, 4>>>();

    for (size_t k = 0; k < data.size() - 1; ++k) {
      auto& [t_k, u_k, p_k, v_k] = data[k];
      auto& [t_k1, u_k1, p_k1, v_k1] = data[k + 1];

      T += t_k1 - t_k;
      ++elems;
    }
  }
  T /= static_cast<double>(elems);

  sleipnir::OptimizationProblem problem;

  auto A = problem.DecisionVariable(2, 2);
  auto B = problem.DecisionVariable(2, 1);
  auto c = problem.DecisionVariable(2, 1);

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

      // See equation (2.11) of
      // https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.pdf
      Eigen::Matrix<double, 2, 2> Q{
          {1.0 / std::pow(positionStddev.value(), 2), 0.0},
          {0.0, 1.0 / std::pow(velocityStddev.value(), 2)}};

      J += (x_next - (A * x + B * u + c * sign(v_k))).T() * Q *
           (x_next - (A * x + B * u + c * sign(v_k)));
    }
  }
  problem.Minimize(J);

  //          A         B       c
  //        [1  ?]     [0]     [0]
  // xₖ₊₁ = [0  ?]xₖ + [?]uₖ + [?]sgn(xₖ)
  problem.SubjectTo(A(0, 0) == 1);
  problem.SubjectTo(A(1, 0) == 0);
  problem.SubjectTo(B(0, 0) == 0);
  problem.SubjectTo(c(0, 0) == 0);

  problem.Solve();

  //              A           B          c
  //         [0     1  ]    [ 0  ]    [  0   ]
  // dx/dt = [0  -Kv/Ka]x + [1/Ka]u + [-Ks/Ka]sgn(x)
  Eigen::Matrix<double, States, States> contA;
  Eigen::Matrix<double, States, Inputs> contB;
  Eigen::Matrix<double, States, Inputs> contC;
  UndiscretizeAB(Eigen::Matrix<double, 2, 2>{A.Value()},
                 Eigen::Matrix<double, 2, 1>{B.Value()}, units::second_t{T},
                 &contA, &contB);
  UndiscretizeAB(Eigen::Matrix<double, 2, 2>{A.Value()},
                 Eigen::Matrix<double, 2, 1>{c.Value()}, units::second_t{T},
                 &contA, &contC);

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
      sleipnir::VariableMatrix A{States, States};
      A(0, 0) = 0;
      A(0, 1) = 1;
      A(1, 0) = 0;
      A(1, 1) = -Kv / Ka;
      sleipnir::VariableMatrix B{States, Inputs};
      B(0, 0) = 0;
      B(1, 0) = 1 / Ka;
      sleipnir::VariableMatrix c{States, 1};
      c(0, 0) = 0;
      c(1, 0) = -Ks / Ka * sign(v_k);

      // Discretize model without B so it can be reused for c
      sleipnir::VariableMatrix M{States + Inputs, States + Inputs};
      M.Block(0, 0, States, States) = A;
      for (int row = 0; row < std::min(States, Inputs); ++row) {
        M(row, States + row) = sleipnir::Constant(1.0);
      }
      sleipnir::VariableMatrix phi = expm(M * T);

      sleipnir::VariableMatrix A_d = phi.Block(0, 0, States, States);
      sleipnir::VariableMatrix B_d = phi.Block(0, States, States, Inputs) * B;
      sleipnir::VariableMatrix c_d = phi.Block(0, States, States, Inputs) * c;
      auto f = [&](const auto& x, const auto& u) {
        return A_d * x + B_d * u + c_d;
      };

      Eigen::Matrix<double, 2, 2> Q{
          {1.0 / std::pow(positionStddev.value(), 2), 0.0},
          {0.0, 1.0 / std::pow(velocityStddev.value(), 2)}};

      J += (x_next - f(x, u)).T() * Q * (x_next - f(x, u));
    }
  }
  problem.Minimize(J);

  problem.Solve();

  return {Ks.Value(), Kv.Value(), Ka.Value()};
}

int main(int argc, const char* argv[]) {
  constexpr units::meters_per_second_t kMotionThreshold = 0.1_mps;
  constexpr units::meter_t kPositionStddev = 0.1_m;
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
    wpi::raw_fd_istream is{args[1], ec};
    if (ec) {
      fmt::print(stderr, "Failed to open file '{}'\n", args[1]);
      return 1;
    }

    is >> json;
  }

  auto startTime = std::chrono::system_clock::now();
  auto initialGuessEigenSysIdOLS = SolveEigenSysIdOLS(json, kMotionThreshold);
  auto endTime = std::chrono::system_clock::now();

  fmt::print("Eigen SysId OLS (velocity only)\n");
  fmt::print("  duration = {} ms\n", ToMilliseconds(endTime - startTime));
  fmt::print("  Ks = {}\n", initialGuessEigenSysIdOLS.Ks);
  fmt::print("  Kv = {}\n", initialGuessEigenSysIdOLS.Kv);
  fmt::print("  Ka = {}\n", initialGuessEigenSysIdOLS.Ka);

  startTime = std::chrono::system_clock::now();
  auto initialGuessSleipnirSysIdOLS =
      SolveSleipnirSysIdOLS(json, kMotionThreshold);
  endTime = std::chrono::system_clock::now();

  fmt::print("Sleipnir SysId OLS (velocity only)\n");
  fmt::print("  duration = {} ms\n", ToMilliseconds(endTime - startTime));
  fmt::print("  Ks = {}\n", initialGuessSleipnirSysIdOLS.Ks);
  fmt::print("  Kv = {}\n", initialGuessSleipnirSysIdOLS.Kv);
  fmt::print("  Ka = {}\n", initialGuessSleipnirSysIdOLS.Ka);

  startTime = std::chrono::system_clock::now();
  auto initialGuessSleipnirLinearSystem = SolveSleipnirLinearSystem(
      json, kMotionThreshold, kPositionStddev, kVelocityStddev);
  endTime = std::chrono::system_clock::now();

  fmt::print("Sleipnir LinearSystem (position and velocity)\n");
  fmt::print("  duration = {} ms\n", ToMilliseconds(endTime - startTime));
  fmt::print("  Ks = {}\n", initialGuessSleipnirLinearSystem.Ks);
  fmt::print("  Kv = {}\n", initialGuessSleipnirLinearSystem.Kv);
  fmt::print("  Ka = {}\n", initialGuessSleipnirLinearSystem.Ka);

  startTime = std::chrono::system_clock::now();
  auto gains =
      SolveSleipnirNonlinear(json, kMotionThreshold, kPositionStddev,
                             kVelocityStddev, initialGuessSleipnirLinearSystem);
  endTime = std::chrono::system_clock::now();

  fmt::print("Sleipnir nonlinear (position and velocity)\n");
  fmt::print("  duration = {} ms\n", ToMilliseconds(endTime - startTime));
  fmt::print("  Ks = {}\n", gains.Ks);
  fmt::print("  Kv = {}\n", gains.Kv);
  fmt::print("  Ka = {}\n", gains.Ka);
}
