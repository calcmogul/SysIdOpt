#!/usr/bin/env python3

import json
import sys

import casadi as ca


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def main():
    if len(sys.argv) == 1:
        print("Specify a JSON filename.")
        sys.exit(1)

    # See https://github.com/wpilibsuite/sysid/blob/main/docs/data-collection.md
    # Non-Drivetrain Mechanisms:
    #   timestamp, voltage, position, velocity
    with open(sys.argv[1]) as f:
        json_data = json.load(f)
        ts = []
        xs = []
        us = []
        for test_name in [
            "fast-backward",
            "fast-forward",
            "slow-backward",
            "slow-forward",
        ]:
            ts += [sample[0] for sample in json_data[test_name]]
            us += [sample[1] for sample in json_data[test_name]]
            xs += [sample[3] for sample in json_data[test_name]]

    solver = ca.Opti()

    Ks = solver.variable()
    Kv = solver.variable()
    Ka = solver.variable()

    J = 0
    for k in range(len(ts) - 1):
        T = ts[k + 1] - ts[k]

        # dx/dt = Ax + Bu + c
        # xₖ₊₁ = eᴬᵀxₖ + A⁻¹(eᴬᵀ − 1)(Buₖ + c)
        # xₖ₊₁ = A_d xₖ + A⁻¹(A_d − 1)(Buₖ + c)
        A = -Kv / Ka
        B = 1 / Ka
        c = -Ks / Ka * sign(xs[k])
        A_d = ca.exp(A * T)
        f = lambda x, u: A_d * x + 1 / A * (A_d - 1) * (B * u + c)

        J += (xs[k + 1] - f(xs[k], us[k])) ** 2
    solver.minimize(J)

    solver.set_initial(Ks, 1)
    solver.set_initial(Kv, 1)
    solver.set_initial(Ka, 1)

    solver.solver("ipopt")
    sol = solver.solve()

    print(f"Ks = {sol.value(Ks)}")
    print(f"Kv = {sol.value(Kv)}")
    print(f"Ka = {sol.value(Ka)}")


if __name__ == "__main__":
    main()
