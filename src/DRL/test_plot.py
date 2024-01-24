import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

class LiveQValuePlotter:
    def __init__(self, maxlen=20):
        self.q_values = deque(maxlen=maxlen)
        self.fig, self.ax = plt.subplots()

    def add_q_values(self, q_values):
        self.q_values.append(q_values)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        if self.q_values:
            data = np.array(self.q_values)
            for i in range(data.shape[1]):
                self.ax.plot(data[:, i], label=f'Q{i}')
            self.ax.legend(loc='upper left')
            self.ax.set_title('Q-Values Over Time')
            self.ax.set_xlabel('Step')
            self.ax.set_ylabel('Q-Value')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
def main():
    plt.ion()
    q_value_plotter = LiveQValuePlotter(maxlen=20)

    for i in range(1000):
        time.sleep(0.1)
        fake_q_values = [random.random() for _ in range(4)]  # pretend Q-values
        q_value_plotter.add_q_values(fake_q_values)  # Update plot with new Q-values

if __name__ == "__main__":
    main()