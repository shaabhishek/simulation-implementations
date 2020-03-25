from math import log, pow, gamma

import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter
from distributions.utils import get_parser, test_mean

class Gamma(Distribution):
    """Law page 453"""
    def __init__(self, alpha:float, lamb:float, ugen:RngStream):
        self.alpha = alpha
        self.lamb = lamb
        self.ugen = ugen

    @property
    def mean(self):
        return self.alpha / self.lamb

    @property
    def var(self):
        return self.alpha / self.lamb**2

    def _sample(self):
        raise NotImplementedError

def setup_distribution():
    ugen = RngStream()
    dist = Gamma(alpha=1.5, lamb=6, ugen=ugen)
    return dist

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    num_samples = args.num_samples

    dist = setup_distribution()

    samples = dist.sample(num_samples)
    DensityPlotter(dist=dist).make_plots(samples.flatten())
    test_mean(dist, samples)