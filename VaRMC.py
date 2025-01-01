"""Value at Risk using Monte-Carlo."""

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore


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


class VaRMC:

    def __init__(self, S, mu, sig, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sig = sig
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulate(self):
        rand = np.sqrt(self.n) * np.random.normal(0, 1, self.iterations)
        simulated_prices = self.S * np.exp(
            self.n * (self.mu - 0.5 * (self.sig**2)) + self.sig * rand
        )
        stock_prices = np.sort(simulated_prices)
        percentile = np.percentile(stock_prices, (1 - self.c) * 100)
        return self.S - percentile


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
    varMC = VaRMC(pos, mu, sig, c, N, 1000)
    print(
        f"VaR from MC at {100*c}% confidence for"
        f" {N} days: ${varMC.simulate():.2f}"  # noqa: E501
    )
