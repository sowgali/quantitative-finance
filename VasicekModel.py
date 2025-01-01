"""Vasicek Model using Monte-Carlo."""

import matplotlib.pyplot as plt
import numpy as np


def vasicek_model(
    r0: float,
    kappa: float,
    theta: float,
    sig: float,
    T: float = 1.0,
    N: int = 1000,  # noqa: E501
):
    """Simulate Vasecik Model using MC.

    Parameters
    ----------
    r0 : float
        Inital interest rate
    kappa : float
        Speed of mean-reversion
    theta : float
        Mean of interest rate
    sig : float
        Volatility
    T : float, optional
        Time period, by default 1.0
    N : int, optional
        Number of simulations, by default 1000
    """
    x = np.zeros(N)
    x[0] = r0
    dt = T / N
    t = np.linspace(0, T, N)

    for i in range(1, N):
        x[i] = (
            x[i - 1]
            + kappa * (theta - x[i - 1]) * dt
            + np.sqrt(dt) * sig * np.random.normal(0, 1)  # noqa: E501
        )

    return t, x


def plot_rates(data, t):
    """Plots the interest rate."""
    plt.plot(t, data)
    plt.xlabel("t")
    plt.ylabel("Interest Rate r(t)")
    plt.title("Vasicek Model")
    plt.show()


if __name__ == "__main__":
    t, data = vasicek_model(1.3, 0.9, 1.5, 0.9)
    plot_rates(data, t)
