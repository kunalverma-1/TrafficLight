import matplotlib.pyplot as plt


def plot_waiting_time(waiting_times):

    plt.figure(figsize=(8, 5))
    plt.plot(waiting_times)

    plt.title("Waiting Time Analysis")
    plt.xlabel("Simulation Step")
    plt.ylabel("Waiting Time")

    plt.grid(True)

    plt.savefig("plots/waiting_time_analysis.png")
    