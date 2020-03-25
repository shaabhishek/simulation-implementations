from math import log, sin, cos, sqrt, pi

import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter
from distributions.utils import get_parser, test_mean

class Gaussian(Distribution):
    """Gaussian using the Box-Mueller transform (Law page 457)"""
    def __init__(self, mu:float, sigma:float, ugen1:RngStream, ugen2:RngStream):
        self._mu = mu
        self._sigma = sigma
        self._var = sigma**2
        self.ugen1 = ugen1
        self.ugen2 = ugen2
        self.cache_size = 100000 #TODO expose this
        self.init_cache()

    @property
    def mean(self):
        return self._mu

    @property
    def var(self):
        return self._var
        
    def init_cache(self):
        self.cache = []

    def fill_cache(self):
        l = len(self.cache)
        u = np.stack([[self.ugen1.RandU01(), self.ugen2.RandU01()] for _ in range(self.cache_size//2)], axis=0)
        t = np.sqrt(-2*np.log(u[:,0]))
        newvals = list(np.stack([t*np.cos(2*np.pi*u[:,1]), t*np.sin(2*np.pi*u[:,1])], axis=1).flatten())

        # Saved as a reversed list on the left, so that pop can be used to sample
        self.cache = newvals[::-1] + self.cache
        print(f"Filled cache. Old size: {l}, New size: {len(self.cache)}")
    
    def _sample(self):
        try:
            return self.cache.pop()*self._sigma + self._mu
        except IndexError:
            self.fill_cache()
            return self.cache.pop()*self._sigma + self._mu

class StdNormal(Gaussian):
    def __init__(self, ugen1:RngStream, ugen2:RngStream):
        super().__init__(0, 1, ugen1, ugen2)

    def _sample(self):
        try:
            return self.cache.pop()
        except IndexError:
            self.fill_cache()
            return self.cache.pop()

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    num_samples = args.num_samples

    ugen1 = RngStream()
    ugen2 = RngStream()
    normal_dist = StdNormal(ugen1=ugen1, ugen2=ugen2)
    # normal_dist = Gaussian(mu=10, sigma=1, ugen1=ugen1, ugen2=ugen2)

    samples = normal_dist.sample(num_samples)

    DensityPlotter(dist=normal_dist).make_plots(samples.flatten())
    test_mean(normal_dist, samples)