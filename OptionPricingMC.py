"""Option pricing using Monte Carlo."""

import numpy as np


class OptionPricing:

    def __init__(self, S0, E, T, rf, sig, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sig = sig
        self.iterations = iterations

    def call_option_price(self):
        rand = np.sqrt(self.T) * np.random.normal(0, 1, self.iterations)

        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * (self.sig**2)) + self.sig * rand
        )

        price_change = stock_price - self.E
        option_price = np.mean(np.where(price_change > 0, price_change, 0))

        return option_price

    def put_option_price(self):
        rand = np.sqrt(self.T) * np.random.normal(0, 1, self.iterations)

        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * (self.sig**2)) + self.sig * rand
        )

        price_change = self.E - stock_price
        option_price = np.mean(np.where(price_change > 0, price_change, 0))

        return option_price


if __name__ == "__main__":
    op = OptionPricing(100, 100, 1, 0.05, 0.2, 1000)
    print(f"Call option price: ${op.call_option_price():.2f}")
    print(f"Put option price: ${op.put_option_price():.2f}")
