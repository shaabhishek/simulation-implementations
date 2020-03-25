import argparse

import numpy as np

from distributions.distribution import Distribution

def get_parser():
    parser = argparse.ArgumentParser(description="Random Number Generators")
    parser.add_argument("-d", "--dist", choices=['exp'], help="Distribution to simulate")
    parser.add_argument("-N", "--num_samples", default=1000, type=int, help="Number of samples to compute density")
    return parser

def test_mean(dist:Distribution, samples:np.ndarray):
    print(f"True mean: {dist.mean:.3f}, Empirical mean: {samples.mean():.3f}")