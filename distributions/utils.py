import argparse

import numpy as np

from distributions.distribution import Distribution

def get_parser():
    parser = argparse.ArgumentParser(description="Random Number Generators")
    parser.add_argument("-d", "--dist", choices=['exp'], help="Distribution to simulate")
    parser.add_argument("-N", "--num_samples", default=1000, type=int, help="Number of samples to compute density")
    parser.add_argument("-p", "--plot_density", action='store_true', help="Flag to show density plot")
    return parser

def test_mean(dist:Distribution, samples:np.ndarray):
    print(f"True mean: {dist.mean:.3f}, Empirical mean: {samples.mean():.3f}")

def test_var(dist:Distribution, samples:np.ndarray):
    print(f"True var: {dist.var:.3f}, Empirical var: {samples.var():.3f}")