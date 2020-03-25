import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter
from distributions.utils import get_parser, test_mean

class SymmetricTriangular(Distribution):
    def __init__(self, low:float, high:float, ugen1:RngStream, ugen2:RngStream):
        self.low = low
        self.high = high
        self.width = high-low
        self._mean = (low+high)/2
        self._var = ((self.width/2)**2) / 6
        self.ugen1 = ugen1
        self.ugen2 = ugen2

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    def _sample(self):
        u1, u2 = self.ugen1.RandU01(), self.ugen2.RandU01()
        return self.low + (u1 + u2) * (self.width/2)

def setup_distribution():
    ugen1 = RngStream()
    ugen2 = RngStream()
    # symm_tri_dist = SymmetricTriangular(ugen1=ugen1, ugen2=ugen2)
    symm_tri_dist = SymmetricTriangular(low=1, high=10, ugen1=ugen1, ugen2=ugen2)
    # symm_tri_dist = Gaussian(mu=10, sigma=1, ugen1=ugen1, ugen2=ugen2)
    return symm_tri_dist

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    num_samples = args.num_samples

    symm_tri_dist = setup_distribution()

    samples = symm_tri_dist.sample(num_samples)
    DensityPlotter(dist=symm_tri_dist).make_plots(samples.flatten())
    test_mean(symm_tri_dist, samples)