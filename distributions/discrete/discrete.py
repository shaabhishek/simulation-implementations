import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter
from distributions.utils import get_parser, test_mean, test_var

class Discrete(Distribution):
    def __init__(self, pi:list, ugen:RngStream):
        assert len(np.shape(pi)) == 1
        self.pi = np.array(pi)
        self.pi /= self.pi.sum() #normalize
        self.cdf = np.cumsum(self.pi)
        self.k = len(pi)
        self.ugen = ugen

    @property
    def mean(self):
        return (np.arange(self.k) * self.pi).sum()

    @property
    def var(self):
        E_x = self.mean
        E_xsquared = (np.arange(self.k)**2 * self.pi).sum()
        return E_xsquared - E_x**2

    def _sample(self):
        u = self.ugen.RandU01()
        for _k in range(self.k):
            if u < self.cdf[_k]:
                return _k


def setup_distribution():
    ugen = RngStream()
    dist = Discrete(pi=[.7, .1, .2], ugen=ugen)
    return dist

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    num_samples = args.num_samples

    dist = setup_distribution()

    samples = dist.sample(num_samples)
    
    # Plotting
    if args.plot_density:
        DensityPlotter(dist=dist).make_plots(samples.flatten())

    # Testing
    test_mean(dist, samples)
    test_var(dist, samples)
