"""Bond Price using Vasicek Model."""

from copy import deepcopy

import numpy as np

NUM_OF_SIMULATIONS = 1000

NUM_OF_POINTS = 200


def monte_carlo_simulation(x, r0, kappa, theta, sig, T=1.0):
    """Simulate MC using Vasicek."""
    dt = T / float(NUM_OF_POINTS)
    result = np.zeros((NUM_OF_SIMULATIONS, NUM_OF_POINTS))
    for i in range(NUM_OF_SIMULATIONS):
        rates = np.zeros(NUM_OF_POINTS)
        rates[0] = r0
        for j in range(1, NUM_OF_POINTS):
            rates[j] = (
                rates[j - 1]
                + kappa * (theta - rates[j - 1]) * dt
                + np.sqrt(dt) * sig * np.random.normal(0, 1)  # noqa: E501
            )
        result[i] = deepcopy(rates)
    integral_sum = result.sum(axis=1) * dt
    bond_price = x * np.mean(np.exp(-integral_sum))
    return bond_price


if __name__ == "__main__":
    bond_price = monte_carlo_simulation(1000, 0.1, 0.3, 0.3, 0.03)
    print(f"Value of the bond is: ${bond_price:.2f}")
