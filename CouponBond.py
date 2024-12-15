"""Coupon bond implementation."""


class CouponBond:
    def __init__(
        self,
        principal: float,
        interest: float,
        maturity: int,
        market_rate: float,  # noqa: E501
    ):
        """Construct a coupon bond.

        Parameters
        ----------
        principal : float
            Face value of the coupon bond
        interest : float
            Interest of bond for coupons
        maturity : int
            Time to maturiy in years
        market_rate : float
            Market interest rate
        """
        self.principal = principal
        self.interest = interest / 100
        self.coupon = self.principal * self.interest
        self.maturity = maturity
        self.market_rate = market_rate / 100

    def present_value(self, current_time: int) -> float:
        """Calculate present value of the bond.

        Parameters
        ----------
        current_time : int
            Current time period

        Returns
        -------
        float
            Present value of the bond
        """
        discount_factor = 1.0 / (1 + self.market_rate)
        coupons = 0.0
        for i in range(current_time + 1, self.maturity + 1):
            coupons += (self.coupon) * (discount_factor ** (i - current_time))
        return (
            self.principal
            * (discount_factor ** (self.maturity - current_time))  # noqa: E501
        ) + coupons


if __name__ == "__main__":
    bond = CouponBond(1000, 10, 3, 4)
    print(
        f"Value of 3-yr bond with 10% coupon and $1000 face value"
        f" is: {bond.present_value(0):.2f}"
    )
