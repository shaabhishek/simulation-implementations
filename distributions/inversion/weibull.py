from math import log, pow, gamma

import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter
from distributions.utils import get_parser, test_mean


class Weibull(Distribution):
    """Law page 456"""
    def __init__(self, alpha:float, beta:float, ugen:RngStream):
        self.alpha = alpha
        self.beta = beta
        self.invalpha = 1/alpha
        self.ugen = ugen

    @property
    def mean(self):
        return gamma(self.invalpha) * self.invalpha * self.beta

    @property
    def var(self):
        return self.beta**2 * self.invalpha * (2*gamma(2*self.invalpha) - self.invalpha*gamma(self.invalpha)**2)

    def _sample(self):
        u = self.ugen.RandU01()
        return self.beta * pow(-log(u), self.invalpha)

def setup_distribution():
    ugen = RngStream()
    dist = Weibull(alpha=1.5, beta=6, ugen=ugen)
    return dist

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    num_samples = args.num_samples

    dist = setup_distribution()

    samples = dist.sample(num_samples)
    DensityPlotter(dist=dist).make_plots(samples.flatten())
    test_mean(dist, samples)