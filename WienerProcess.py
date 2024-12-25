import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr


def wiener_process(dt=0.1, x_0=0, n=10000):
    """Simulate Wiener process."""
    time_data = np.linspace(x_0, n, n + 1)
    wiener_data = np.zeros(n + 1)
    wiener_data[1:] = np.cumsum(npr.normal(0, np.sqrt(dt), n))
    return wiener_data, time_data


def plot_data(W, t):
    """Plot Wiener Process data."""
    plt.plot(t, W)
    plt.xlabel("Time(t)")
    plt.ylabel("Wiener Process W(t)")
    plt.title("Wiener process")
    plt.show()


if __name__ == "__main__":
    W, t = wiener_process()
    plot_data(W, t)
