"""Implementation of zero coupon bond."""


class ZeroCouponBond:

    def __init__(self, amount: float, maturity: int, market_rate: float):
        """Construct a zcp.

        Parameters
        ----------
        amount : float
            Face Value of the bond
        maturity : int
            Maturity time in years
        market_rate : float
            Current market interest rate
        """
        self.amount = amount
        self.maturity = maturity
        self.market_rate = market_rate / 100

    def present_value(self, current_period: int) -> float:
        """Calculate present value of the bond.

        Parameters
        ----------
        current_period : float
            Current year, e.g, maturity = 5
            we are in year 2, remaining 3

        Returns
        -------
        float
            Current bond rate
        """
        return self.amount / (1.0 + self.market_rate) ** (
            self.maturity - current_period
        )


if __name__ == "__main__":
    zcpBond = ZeroCouponBond(1000, 5, 4)
    print(f"Current bond price: {zcpBond.present_value(3):.2f}")
