import matplotlib.pyplot as plt

def plot_training(iterations: list, rewards: list, figpath: str):
    # Feel free to use the space below to run experiments and plots used in your writeup.
    plt.figure(figsize=(10,5))
    plt.title("Rewards vs iterations")
    plt.plot(iterations, rewards)
    plt.xlabel("Iterations")
    plt.ylabel("Rewards")
    plt.grid()
    plt.savefig(f'{figpath}.png')