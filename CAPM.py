import matplotlib.pyplot as plt
import numpy as np  # type: ignore
import pandas as pd
import yfinance as yf  # type: ignore

RISK_FREE_RETURN = 0.05
MONTHS_IN_YEAR = 12


class CAPM:

    def __init__(self, stocks, start_date, end_date):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}
        for stock in self.stocks:
            ticker = yf.download(
                stock,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                period="1mo",
            )
            data[stock] = ticker["Close"][stock]
        return pd.DataFrame(data)

    def initialize(self):
        stock_data = self.download_data()

        self.data = pd.DataFrame(
            {
                "s_adjclose": stock_data[self.stocks[0]],
                "m_adjclose": stock_data[self.stocks[1]],
            }
        )
        self.data[["s_returns", "m_returns"]] = np.log(
            self.data[["s_adjclose", "m_adjclose"]]
            / self.data[["s_adjclose", "m_adjclose"]].shift(1)
        )
        self.data = self.data[1:]

    def calculate_beta(self):
        beta = (
            self.data["s_returns"].cov(self.data["m_returns"])
            / self.data["m_returns"].var()
        )
        print(f"Beta from calculation: {beta}")

    def regression(self):
        # Re = α + β Rm
        beta, alpha = np.polyfit(
            x=self.data["m_returns"], y=self.data["s_returns"], deg=1
        )
        print(f"Beta from regression: {beta}")
        expected_return = RISK_FREE_RETURN + beta * (
            self.data["m_returns"].mean() * MONTHS_IN_YEAR - RISK_FREE_RETURN
        )
        print(f"Expected return: {expected_return}")
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        _, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(
            self.data["m_returns"], self.data["s_returns"], label="Data Points"
        )
        axis.plot(
            self.data["m_returns"],
            beta * self.data["m_returns"] + alpha,
            color="red",
            label="CAPM line",
        )
        plt.title("Capital Asset Pricing Model, finding alpha and beta")
        plt.xlabel("Market return $R_m$", fontsize=18)
        plt.ylabel("Stock return $R_a$")
        plt.text(0.08, 0.05, r"$R_a = \beta * R_m + \alpha$", fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    capm = CAPM(["IBM", "^GSPC"], "2012-01-01", "2024-12-01")
    capm.initialize()
    capm.calculate_beta()
    capm.regression()
