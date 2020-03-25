import numpy as np
import matplotlib.pyplot as plt

from distributions.distribution import Distribution

class DistPlotter():
    def __init__(self, dist:Distribution):
        self.dist = dist

class SamplingDistPlotter(DistPlotter):
    def __init__(self, dist:Distribution):
        super().__init__(dist)

    def plot_sampling_distribution(self, num_repl:int, N:int, ax:plt.Axes):

        sampling_dist_samples = []
        for _ in range(num_repl):
            sample = [self.dist.sample() for _ in range(N)]
            sampling_dist_samples.append(sample)

        print(f"Empirical mean: {np.mean(sampling_dist_samples)}, True mean: {self.dist.mean}")
        print(f"Empirical variance: {np.var(sampling_dist_samples, ddof=1)}, True variance: {self.dist.var}")

        ax.hist(np.mean(sampling_dist_samples, axis=1), bins=100)
        ax.set_title("Sampling distribution")

    def make_plots(self, num_repl:int, N:int):
        _, ax = plt.subplots(1,1)
        self.plot_sampling_distribution(num_repl=num_repl, N=N, ax=ax)

        plt.show()

class DensityPlotter(DistPlotter):
    def __init__(self, dist:Distribution):
        super().__init__(dist)

    def plot_density(self, num_samples:int, ax:plt.Axes):
        samples = [self.dist.sample() for _ in range(num_samples)]
        ax.hist(samples, bins=100)
        ax.set_title("Density")

    def make_plots(self, num_samples:int):
        _, ax = plt.subplots(1,1)
        self.plot_density(num_samples, ax)

        plt.show()