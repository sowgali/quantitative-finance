"""Implementation of Mean-Markowitz Model."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt  # type: ignore
import yfinance as yf  # type: ignore

NUM_TRADING_DAYS = 252

NUM_PORTFOLIOS = 10000

tickers = ["AAPL", "TSLA", "AMZN", "MA", "DG"]

start_date = "2019-01-01"
end_date = "2024-12-10"


def download_data() -> pd.DataFrame:
    """Download ticker data for the listed tickers.

    Returns
    -------
    pd.DataFrame
        Ticker dataframe
    """
    stock_data = {}

    for stock in tickers:
        ticker = yf.Ticker(stock)
        history = ticker.history(start=start_date, end=end_date)
        stock_data[stock] = history["Close"]

    return pd.DataFrame(stock_data)


def calculate_returns(data: pd.DataFrame):
    """Calculate log returns of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Stock Data

    Returns
    -------
    pd.DataFrame
        Log returns of stock
    """
    return np.log(data / data.shift(1))[1:]


def show_statistics(data: pd.DataFrame):
    """Show statistics for the stock data.

    Parameters
    ----------
    data : pd.DataFrame
        stock data

    Returns
    -------
    pd.DataFrame
        stock data statistics
    """
    print(data.mean() * NUM_TRADING_DAYS)
    print(data.cov() * NUM_TRADING_DAYS)


def statistics(weights, returns):
    """Get statistics for this portfolio."""
    portfolio_returns = np.sum(weights * returns.mean()) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov(), weights)) * NUM_TRADING_DAYS
    )
    sharpe_ratio = portfolio_returns / portfolio_volatility
    return np.array([portfolio_returns, portfolio_volatility, sharpe_ratio])


def min_function_sharpe(weights, returns):
    """Min function to optimize for sharpe values."""
    return -statistics(weights, returns)[2]


def show_mean_variance(returns, weights):
    """Shows Mean Variance of the portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Average log retuns
    weights : np.array
        Weights of stocks in portfolio
    """
    portfolio_returns = np.sum(weights * returns.mean()) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov(), weights)) * NUM_TRADING_DAYS
    )
    print(portfolio_returns)
    print(portfolio_volatility)


def show_data(data: pd.DataFrame):
    """Plot the ticker history data."""
    data.plot(figsize=(10, 5))
    plt.show()


def generate_portfolios(returns):
    """Generate portfolios with varying weights."""
    unnormalized_weights = np.random.rand(
        NUM_PORTFOLIOS, returns.shape[1]
    )  # noqa: E501
    normalized_weights = unnormalized_weights / unnormalized_weights.sum(
        axis=1, keepdims=True
    )
    batched_normalized_weights = np.expand_dims(normalized_weights, 2)
    batched_cov = np.repeat(
        np.array(returns.cov() * NUM_TRADING_DAYS).reshape(
            -1, returns.shape[1], returns.shape[1]
        ),
        NUM_PORTFOLIOS,
        axis=0,
    )

    simulated_portfolio_means = np.sum(
        normalized_weights * np.array(returns.mean() * NUM_TRADING_DAYS),
        axis=1,  # noqa: E501
    )
    simulated_portfolio_vol = np.sqrt(
        np.matmul(
            np.transpose(batched_normalized_weights, axes=(0, 2, 1)),
            np.matmul(batched_cov, batched_normalized_weights),
        )
    ).reshape(-1)

    return (
        normalized_weights,
        simulated_portfolio_means,
        simulated_portfolio_vol,
    )  # noqa: E501


def optimize_portfolio(weights, returns):
    """Optimizes the weights to have maximum sharpe ratio."""
    # Sum of weights should be 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    # Weights should be between 0 and 1
    bounds = tuple((0, 1) for _ in range(len(weights[0])))
    return opt.minimize(
        fun=min_function_sharpe,
        x0=weights[0],
        args=returns,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    """Plot optimal portfolio."""
    plt.figure(figsize=(10, 6))
    plt.scatter(
        portfolio_vols,
        portfolio_rets,
        c=portfolio_rets / portfolio_vols,
        marker="o",  # noqa: E501
    )
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.plot(
        statistics(opt["x"], rets)[1],
        statistics(opt["x"], rets)[0],
        "g*",
        markersize=20.0,
    )
    plt.show()


def print_optimal_portfolio(optimum, returns):
    """Print optimal portfolio weights."""
    print(f"Optimal portfolio: {optimum['x'].round(3)}")
    print(
        "expected return, volatility, and Sharpe Ratio: ",
        statistics(optimum["x"].round(3), returns),
    )


def show_portfolios(returns, volatilities):
    """Plot returns and risks of portfolios with Sharpe Ratios."""
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker="o")
    plt.grid(True)
    plt.xlabel("Expected volatility")
    plt.ylabel("Expected return")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()


if __name__ == "__main__":
    data = download_data()
    log_returns = calculate_returns(data)
    # show_statistics(log_returns)
    pweights, returns, volatilities = generate_portfolios(log_returns)
    # show_portfolios(returns, volatilities)
    # show_data(log_returns)
    optimum = optimize_portfolio(pweights, log_returns)
    print_optimal_portfolio(optimum, log_returns)
    show_optimal_portfolio(optimum, log_returns, returns, volatilities)
