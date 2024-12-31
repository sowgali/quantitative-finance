"""Monte-Carlo simulation for Stock Price."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM_OF_SIMULATIONS = 10000


def stock_price_MC(S0: float, mu: float, sig: float, N: int = 252) -> float:
    """Get stock price from Monte-Carlo simulation.

    Parameters
    ----------
    S0 : float
        Current stock price
    mu : float
        Mean growth
    sig : float
        Voltility
    N : int, optional
        Number of days, by default 252 (Trading days in a year)

    Returns
    -------
    float
        Stock prediction from MC simulation
    """
    W = np.random.normal(size=(NUM_OF_SIMULATIONS, N))
    t = np.repeat(
        np.expand_dims(np.arange(start=0, stop=252), axis=0),
        NUM_OF_SIMULATIONS,
        axis=0,  # noqa: E501
    )
    exp_part = (mu - 0.5 * (sig**2)) * t + sig * np.cumsum(W, axis=1)
    prices_simulated = S0 * np.exp(exp_part)
    plt.plot(prices_simulated.T)
    plt.show()
    df = pd.DataFrame(prices_simulated).T
    df["mean"] = df.mean(axis=1)
    return df["mean"].iloc[-1]


if __name__ == "__main__":
    stock_price = stock_price_MC(50, 0.0002, 0.01)
    print(f"Average stock price from MC: ${stock_price:.2f}")
