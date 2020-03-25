from math import log

import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter
from distributions.utils import get_parser, test_mean

class Exponential(Distribution):
    def __init__(self, lamb:float, ugen:RngStream):
        self.lamb = lamb
        self.mean = 1/lamb
        self.var = 1/lamb**2
        self.ugen = ugen

    def _sample(self):
        u = self.ugen.RandU01()
        return -log(u)/self.lamb


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    num_samples = args.num_samples

    ugen = RngStream()
    exp_dist = Exponential(1, ugen=ugen)

    samples = exp_dist.sample(num_samples)

    DensityPlotter(dist=exp_dist).make_plots(samples.flatten())
    test_mean(exp_dist, samples)