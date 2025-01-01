"""Value at Risk implementation."""

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore
from scipy.stats import norm  # type: ignore


def download_data(stock: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download ticker data.

    Parameters
    ----------
    stock : str
        Stock ticker
    start_date : str
        Start Date
    end_date : str
        End Date

    Returns
    -------
    pd.DataFrame
        Stock data
    """
    ticker = yf.Ticker(stock)
    stock_data = ticker.history(start=start_date, end=end_date)
    return pd.DataFrame(stock_data["Close"])


def calculate_var(
    pos: float, c: float, mu: float, sig: float, N: int
) -> float:  # noqa: E501
    """Calculates Value at Risk.

    Parameters
    ----------
    pos : float
        Value of the position
    c : float
        Confidence level
    mu : float
        Mean returns
    sig : float
        Volatility
    N : int
        Total period

    Returns
    -------
    float
        Value at Risk
    """
    return pos * (mu * N - sig * np.sqrt(N) * norm.ppf(1.0 - c))


if __name__ == "__main__":
    stock = "C"
    start_date = "2022-01-02"
    end_date = "2024-12-01"
    stock_data = download_data(stock, start_date, end_date)
    log_return = np.log(stock_data / stock_data.shift(1))[1:]
    pos = 1e6
    mu = float(np.mean(log_return))
    sig = float(np.std(log_return, axis=0).iloc[0])
    N = 10
    c = 0.95
    print(
        f"Value at Risk at {100*c}% confidence for {N} days"
        f" is: ${calculate_var(pos, c, mu, sig, N):.2f}"
    )
