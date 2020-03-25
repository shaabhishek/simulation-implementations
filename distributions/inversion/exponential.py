import argparse
from math import log

import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter

class Exponential(Distribution):
    def __init__(self, lamb:float, ugen:RngStream):
        self.lamb = lamb
        self.mean = 1/lamb
        self.var = 1/lamb**2
        self.ugen = ugen

    def sample(self):
        u = self.ugen.RandU01()
        return -log(u)/self.lamb


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Random Number Generators")
    parser.add_argument("-d", "--dist", choices=['exp'], help="Distribution to simulate")
    parser.add_argument("--num_repl", default=1000, type=int, help="Number of samples in the sampling distribution")
    parser.add_argument("-N", "--samples_per_statistic", default=1000, type=int, help="Number of samples to compute statistic in the sampling distribution")
    args = parser.parse_args()

    num_repl = args.num_repl
    N = args.samples_per_statistic

    ugen = RngStream()
    exp_dist = Exponential(1, ugen=ugen)

    DensityPlotter(dist=exp_dist).make_plots(num_repl)