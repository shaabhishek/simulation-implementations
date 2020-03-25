import argparse
from math import log

import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter

class Uniform(Distribution):
    def __init__(self, a:float, b:float, ugen:RngStream):
        self.a = a
        self.b = b
        self.width = b-a
        self.mean = (a+b)/2
        self.var = self.width**2 / 12
        self.ugen = ugen

    def _sample(self):
        u = self.ugen.RandU01()
        return self.a + self.width*u
    
    def sample(self, shape=(1,)):
        num_samples = np.prod(shape)
        samples = [self._sample() for _ in range(num_samples)]
        samples = np.reshape(samples, shape)
        return samples

class Uniform01(Uniform):
    def __init__(self, ugen:RngStream):
        super().__init__(0,1, ugen)

    def sample(self):
        return self.ugen.RandU01()

def get_parser():
    parser = argparse.ArgumentParser(description="Random Number Generators")
    parser.add_argument("-d", "--dist", choices=['exp'], help="Distribution to simulate")
    parser.add_argument("-N", "--num_samples", default=1000, type=int, help="Number of samples to compute density")
    return parser

def test_mean(dist:Distribution, samples:np.ndarray):
    print(f"True mean: {dist.mean:.3f}, Empirical mean: {samples.mean():.3f}")

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    num_samples = args.num_samples

    ugen = RngStream()
    # unif_dist = Uniform01(ugen=ugen)
    unif_dist = Uniform(100, 500, ugen=ugen)

    samples = unif_dist.sample(num_samples)

    DensityPlotter(dist=unif_dist).make_plots(samples)
    test_mean(unif_dist, samples)