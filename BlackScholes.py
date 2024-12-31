"""Black-Scholes solution implementation."""

import numpy as np
from scipy import stats  # type: ignore


def call_option_price(
    S0: float, E: float, sig: float, T: float, rf: float
) -> float:  # noqa: E501
    """Get call option price.

    Parameters
    ----------
    S0 : float
        Initial price of stock
    E : float
        Strike price at expiry
    sig : float
        Volatility of the stock
    T : float
        Time to expiry
    rf : float
        Risk free return

    Returns
    -------
    float
        Call option price
    """
    d1 = (np.log(S0 / E) + (rf + 0.5 * (sig**2)) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S0 * stats.norm.cdf(d1) - E * np.exp(-rf * T) * stats.norm.cdf(d2)


def put_option_price(
    S0: float, E: float, sig: float, T: float, rf: float
) -> float:  # noqa: E501
    """Get put option price.

    Parameters
    ----------
    S0 : float
        Initial price of stock
    E : float
        Strike price at expiry
    sig : float
        Volatility of the stock
    T : float
        Time to expiry
    rf : float
        Risk free return

    Returns
    -------
    float
        Put option price
    """
    d1 = (np.log(S0 / E) + (rf + 0.5 * (sig**2)) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return -S0 * stats.norm.cdf(-d1) + E * np.exp(-rf * T) * stats.norm.cdf(
        -d2
    )  # noqa: E501


if __name__ == "__main__":
    S0 = 100.0
    E = 100.0
    T = 1.0
    sig = 0.2
    rf = 0.05
    print(f"Call option price: ${call_option_price(S0, E, sig, T, rf):.2f}")
    print(f"Put option price: ${put_option_price(S0, E, sig, T, rf):.2f}")
