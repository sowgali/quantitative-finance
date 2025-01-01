"""Implementation of Ornstein Uhlenbeck process."""

import matplotlib.pyplot as plt
import numpy as np


def generate_process(dt=0.1, theta=1.2, mu=0.9, sig=0.9, n=10000):
    x = np.zeros(n)

    for i in range(1, n):
        x[i] = (
            x[i - 1]
            + theta * (mu - x[i - 1]) * dt
            + sig * np.random.normal(0, np.sqrt(dt))
        )
    return x


def plot_process(x):
    plt.plot(x)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Ornstein Uhlenbeck process")
    plt.show()


if __name__ == "__main__":
    data = generate_process()
    plot_process(data)
