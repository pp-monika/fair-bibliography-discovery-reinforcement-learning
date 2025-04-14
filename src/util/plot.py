import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, labels, num_episodes, window_size=300):
    """
    Plot moving averages of multiple reward lists.

    Args:
        rewards (list of lists): Each sublist contains episode rewards for one run.
        labels (list of str): Labels for each reward list, shown in the legend.
        num_episodes (int): Number of episodes in each run.
        window_size (int): Window size for moving average.
    """
    plt.figure(figsize=(12, 8))

    for i, reward_list in enumerate(rewards):
        reward_array = np.array(reward_list)

        # Compute moving average
        if len(reward_array) >= window_size:
            moving_avg = np.convolve(reward_array, np.ones(window_size) / window_size, mode='valid')
            x = range(len(moving_avg))
        else:
            moving_avg = reward_array  # fallback if too short
            x = range(len(reward_array))

        label = labels[i] if i < len(labels) else f'Run {i + 1}'
        plt.plot(x, moving_avg, label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title(f"Moving Average of Rewards Over {num_episodes} Episodes")

    min_y = min([
        min(np.convolve(r, np.ones(window_size) / window_size, mode='valid')) 
        if len(r) >= window_size else min(r) 
        for r in rewards
    ])

    max_y = max([
        max(np.convolve(r, np.ones(window_size) / window_size, mode='valid')) 
        if len(r) >= window_size else max(r) 
        for r in rewards
    ])
    plt.ylim(min_y - 0.5, max_y + 0.5)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()