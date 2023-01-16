// Copyright (c) Tyler Veness

#include <array>
#include <cmath>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <fmt/core.h>
#include <frc/fmt/Eigen.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/time.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <wpi/json.h>
#include <wpi/raw_istream.h>

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

struct FeedforwardGains {
  double Ks = 0.0;
  double Kv = 0.0;
  double Ka = 0.0;
};

/**
 * Solves SysId's OLS problem to produce initial guess for nonlinear problem.
 *
 * @param[in] json SysId JSON.
 * @return Initial guess for nonlinear problem.
 */
FeedforwardGains SolveSysIdOLSProblem(const wpi::json& json) {
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
 * Solves linear system ID problem to produce initial guess for nonlinear
 * problem.
 *
 * @param[in] json SysId JSON.
 * @return Initial guess for nonlinear problem.
 */
FeedforwardGains SolveLinearSystemProblem(const wpi::json& json) {
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

      Eigen::Vector<double, States> x{{p_k}, {v_k}};
      Eigen::Vector<double, States> x_next{{p_k1}, {v_k1}};
      Eigen::Vector<double, Inputs> u{u_k};

      J += (x_next - (A * x + B * u + c * sign(v_k))).T() *
           (x_next - (A * x + B * u + c * sign(v_k)));
    }
  }
  problem.Minimize(J);

  problem.Solve();

  //             A            B          c
  //         [0       1]    [ 0  ]    [  0   ]
  // dx/dt = [0  -Kv/Ka]x + [1/Ka]u + [-Ks/Ka]
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
 * Solves nonlinear system ID problem.
 *
 * @param[in] json SysId JSON.
 * @param[in] initialGuess Initial guess from linear problem.
 */
FeedforwardGains SolveNonlinearProblem(const wpi::json& json,
                                       const FeedforwardGains& initialGuess) {
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
      auto& [t_k, u_k, p_k, x_k] = data[k];
      auto& [t_k1, u_k1, p_k1, x_k1] = data[k + 1];

      double T = t_k1 - t_k;

      // xₖ₊₁ = eᴬᵗxₖ + A⁻¹(eᴬᵗ − 1)(Buₖ + c)
      // xₖ₊₁ = A_d xₖ + A⁻¹(A_d − 1)(Buₖ + c)
      auto A = -Kv / Ka;
      auto B = 1 / Ka;
      auto c = Ks * sign(x_k);
      auto A_d = sleipnir::exp(A * T);
      auto f = [&](const auto& x, const auto& u) {
        return A_d * x + 1 / A * (A_d - 1) * (B * u + c);
      };

      J += sleipnir::pow(x_k1 - f(x_k, u_k), 2);
    }
  }
  problem.Minimize(J);

  problem.Solve();

  return {Ks.Value(), Kv.Value(), Ka.Value()};
}

int main(int argc, const char* argv[]) {
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

  auto initialGuess = SolveSysIdOLSProblem(json);
  fmt::print("OLS Ks = {}\n", initialGuess.Ks);
  fmt::print("OLS Kv = {}\n", initialGuess.Kv);
  fmt::print("OLS Ka = {}\n", initialGuess.Ka);

  auto initialGuess2 = SolveLinearSystemProblem(json);
  fmt::print("LinearSystem Ks = {}\n", initialGuess2.Ks);
  fmt::print("LinearSystem Kv = {}\n", initialGuess2.Kv);
  fmt::print("LinearSystem Ka = {}\n", initialGuess2.Ka);

  auto gains = SolveNonlinearProblem(json, initialGuess);
  fmt::print("Nonlinear Ks = {}\n", gains.Ks);
  fmt::print("Nonlinear Kv = {}\n", gains.Kv);
  fmt::print("Nonlinear Ka = {}\n", gains.Ka);
}
